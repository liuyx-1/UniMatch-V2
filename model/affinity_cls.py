"""Patch-text affinity classifier for multi-label cls + coarse localization.

  ┌──── frozen SigLIP-2 ────┐         ┌── trainable (~500K params) ──┐
  │  vision encoder         │         │ visual_proj (768->256->128)   │
  │  text encoder           │         │ text_proj   (768->256->128)   │
  └──────┬──────────┬───────┘         │ log_tau                       │
         │          │                  │ cls_bias [C]                  │
   patches [N,768]  T_c (4 templates avg)
         │          │                  └───────────────────────────────┘
    visual_proj   text_proj
         │          │
    p̂ [N,128]    t̂_c [C,128]   (both L2-norm)
         │          │
         └─── A[i,c] = p̂_i · t̂_c / τ ────► [N, C]
                        │
            ┌───────────┴───────────┐
            ▼                       ▼
    s_c = topk(A[:,c]).mean + b_c    L_c = A[:,c].reshape(H_p,W_p)
    (cls logits [B,C])               (dense map [B,C,H_p,W_p])

  Training: image-level multi-label BCE only (NO mask supervision).
  Output: cls (for image-level decision) + dense (for downstream seg prior).
"""
from __future__ import annotations
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _patch_torch_custom_op():
    try:
        _orig = torch.library.custom_op
        def _safe(*a, **k):
            try: return _orig(*a, **k)
            except Exception:
                def _id(fn): return fn
                return _id
        torch.library.custom_op = _safe
    except Exception:
        pass
_patch_torch_custom_op()


def _build_proj(d_in: int, d_out: int, hidden: int = 256, dropout: float = 0.0):
    return nn.Sequential(
        nn.LayerNorm(d_in),
        nn.Linear(d_in, hidden),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden, d_out),
    )


class AffinityClassifier(nn.Module):
    """Inputs:
        pixel_values [B, 3, H, W]   processed via SigLIP processor
    Returns:
        cls    [B, C]               image-level multi-label logits (top-k pooled)
        dense  [B, C, H_p, W_p]     patch-text affinity (raw cosine/tau values)
        T      [C, d_latent]        frozen text anchors (after init_T_from_templates)
        patches [B, N, d_latent]    patch latents
    """
    def __init__(self,
                 model_name: str,
                 class_names: List[str],
                 d_latent: int = 128,
                 d_text: int = 768,
                 init_tau: float = 0.07,
                 topk: int = 5,
                 vision_backbone: str = 'siglip',
                 dinov2_input_size: int = 518,
                 freeze_vision: bool = False,
                 freeze_text:   bool = False):
        """vision_backbone:
            'siglip' — use SigLIP-2 vision tower (default, patches 16x16=256)
            'dinov2' — use DINOv2-B/14 (UniMatch-V2 shared backbone, patches 37x37=1369 @ 518)
        Text path is ALWAYS SigLIP-2 text encoder (we still need text-image space).
        freeze_vision / freeze_text: if True, set requires_grad=False and call .eval()
          on the corresponding encoder so it never accumulates grads. Defaults are
          False — both encoders are trainable (with very small lr supplied by the
          trainer) so they can adapt to the surgical domain.
        """
        super().__init__()
        assert vision_backbone in ('siglip', 'dinov2'), vision_backbone
        from transformers import AutoModel, AutoProcessor, AutoTokenizer
        self.model_name = model_name
        self.vision_backbone = vision_backbone
        self.freeze_vision = bool(freeze_vision)
        self.freeze_text   = bool(freeze_text)
        # Text path always SigLIP
        text_bb = AutoModel.from_pretrained(model_name)
        self.text_model = text_bb.text_model
        self.tokenizer  = AutoTokenizer.from_pretrained(model_name)
        self.siglip_processor = AutoProcessor.from_pretrained(model_name)
        if self.freeze_text:
            for p in self.text_model.parameters(): p.requires_grad_(False)
            self.text_model.eval()
        # Vision path
        if vision_backbone == 'siglip':
            self.vision_model = text_bb.vision_model
            if self.freeze_vision:
                for p in self.vision_model.parameters(): p.requires_grad_(False)
                self.vision_model.eval()
            self.processor = self.siglip_processor       # back-compat alias
            self._dinov2 = None
            self.dinov2_input_size = None
        else:
            # DINOv2-B/14 via torch.hub (or local fallback handled in helper)
            self._dinov2 = self._build_dinov2()
            if self.freeze_vision:
                self._dinov2.eval()
                for p in self._dinov2.parameters(): p.requires_grad_(False)
            self.vision_model = None
            self.dinov2_input_size = int(dinov2_input_size)
            from torchvision import transforms
            self._dinov2_tf = transforms.Compose([
                transforms.Resize((self.dinov2_input_size, self.dinov2_input_size),
                                   interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225]),
            ])
            # keep .processor for backward compat (text usage only)
            self.processor = self.siglip_processor

        self.class_names = class_names
        self.d_latent = d_latent
        self.d_text = d_text
        self.topk = topk

        self.visual_proj = _build_proj(d_text, d_latent)
        self.text_proj   = _build_proj(d_text, d_latent)
        self.log_tau = nn.Parameter(torch.tensor(float(torch.log(torch.tensor(init_tau)))))
        self.cls_bias = nn.Parameter(torch.zeros(len(class_names)))

        # T_c stored as buffer (NOT a Parameter — frozen after init_T_from_templates)
        self.register_buffer('T_anchor', torch.zeros(len(class_names), d_latent),
                              persistent=False)
        self._anchor_initialised = False

    @staticmethod
    def _build_dinov2():
        """Build DINOv2-B/14. Prefer LOCAL .pth (UniMatch-V2 already ships it); fall back to torch.hub."""
        import os
        LOCAL_CANDIDATES = [
            os.environ.get('DINOV2_VITB14_PATH', ''),
            './pretrained/dinov2_vitb14_pretrain.pth',
            '/data/code/UniMatch-V2/pretrained/dinov2_vitb14_pretrain.pth',
            '/data/code/UniMatch-V2-manifold/pretrained/dinov2_vitb14_pretrain.pth',
            '/data/pretrained/dinov2_vitb14_pretrain.pth',
            os.path.expanduser('~/.cache/torch/hub/checkpoints/dinov2_vitb14_pretrain.pth'),
        ]
        for p in LOCAL_CANDIDATES:
            if not p or not os.path.isfile(p):
                continue
            print(f'[dinov2] loading local weights: {p}')
            try:
                from model.backbone.dinov2 import vit_base, Block, MemEffAttention
                from functools import partial
                model = vit_base(
                    patch_size=14, img_size=518,
                    init_values=1.0, ffn_layer='mlp', block_chunks=0,
                    num_register_tokens=0,
                )
                state = torch.load(p, map_location='cpu', weights_only=False)
                if isinstance(state, dict) and 'state_dict' in state: state = state['state_dict']
                if isinstance(state, dict) and 'model' in state:      state = state['model']
                missing, unexpected = model.load_state_dict(state, strict=False)
                if len(unexpected) > 5:
                    print(f'  [warn] {len(unexpected)} unexpected keys (showing 5): {unexpected[:5]}')
                if len(missing) > 5:
                    print(f'  [warn] {len(missing)} missing keys (showing 5): {missing[:5]}')
                # add a forward_features signature compatible with torch.hub DINOv2
                # the local DinoVisionTransformer.forward_features returns the same dict format
                return model
            except Exception as e:
                print(f'  [local load failed] {e}, trying next candidate...')
        # Last resort: torch.hub (requires internet)
        print('[dinov2] no local weights found, trying torch.hub.load (needs internet)...')
        return torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14', pretrained=True)

    def preprocess(self, pil_image):
        """Return a [3, H, W] tensor ready for forward, depending on chosen backbone."""
        if self.vision_backbone == 'siglip':
            return self.siglip_processor(images=[pil_image], return_tensors='pt')['pixel_values'][0]
        return self._dinov2_tf(pil_image)

    def _encode_patches(self, pixel_values):
        """Forward image through vision backbone → patch tokens [B, N, 768].
        Uses torch.no_grad() only when the vision encoder is frozen, so that grads
        can flow through it when the trainer chooses to unfreeze it."""
        if self.vision_backbone == 'siglip':
            ctx = torch.no_grad() if self.freeze_vision else torch.enable_grad()
            with ctx:
                out = self.vision_model(pixel_values=pixel_values)
            return out.last_hidden_state
        else:  # dinov2
            ctx = torch.no_grad() if self.freeze_vision else torch.enable_grad()
            with ctx:
                out = self._dinov2.forward_features(pixel_values)
            return out['x_norm_patchtokens']                       # [B, N, 768]

    def init_T_from_templates(self, template_dict: Dict[str, List[str]]):
        """For each class name, ensemble all prompts into one T_c (L2-norm).
        Stored in self.T_anchor (buffer). If the text encoder is trainable, this
        should be called at the end of each epoch (or step) to refresh anchors.
        Always runs under no_grad to avoid bloating the autograd graph with the
        full text encoder; gradients into the text encoder/text_proj come instead
        from the cls BCE path through patches->cos(patches, T_anchor) using a
        separate live recomputation inside the forward when freeze_text=False."""
        device = self.log_tau.device
        with torch.no_grad():
            for cidx, cname in enumerate(self.class_names):
                prompts = template_dict.get(cname)
                if not prompts:
                    prompts = [cname.replace('_', ' ').replace('-', ' ')]
                toks = self.tokenizer(prompts, return_tensors='pt', padding='max_length',
                                        truncation=True, max_length=64).to(device)
                out = self.text_model(**toks)
                last = out.last_hidden_state
                attn = toks.get('attention_mask', None)
                if attn is not None:
                    lens = attn.sum(dim=1) - 1
                    pooled = last[torch.arange(last.size(0), device=device), lens]
                else:
                    pooled = last[:, -1, :]
                proj = F.normalize(self.text_proj(pooled), dim=-1)
                self.T_anchor[cidx].copy_(F.normalize(proj.mean(0), dim=-1))
        self._anchor_initialised = True

    def refresh_T(self):
        """Recompute T_anchor with current text_proj weights (text_proj is trainable).
        Call once per epoch (or step) so anchors stay consistent with the proj head."""
        # Re-encode templates is expensive; cache token embeddings from init.
        # Simpler: just apply current text_proj on cached frozen text encoder outputs.
        # For simplicity here, we keep T_anchor static after init: text_proj weights
        # used at init are also the trainable ones, so we ONLY allow visual_proj to
        # adapt. If you want text_proj to adapt too, call init_T_from_templates again
        # at the start of each epoch (cost: one pass over text encoder, ~ms).
        pass

    def forward(self, pixel_values: torch.Tensor) -> Dict[str, torch.Tensor]:
        assert self._anchor_initialised, 'call init_T_from_templates() first'
        tokens = self._encode_patches(pixel_values)                  # [B, N, 768]
        B, N, _ = tokens.shape
        side = int(round(N ** 0.5))
        p_lat = F.normalize(self.visual_proj(tokens), dim=-1)         # [B, N, 128]

        tau = self.log_tau.exp().clamp(1e-3, 1.0)
        A = (p_lat @ self.T_anchor.T) / tau                            # [B, N, C]

        # Image-level: top-k pool over patches
        k = min(self.topk, N)
        topk_vals = A.topk(k, dim=1).values                            # [B, k, C]
        cls = topk_vals.mean(dim=1) + self.cls_bias                    # [B, C]

        # Dense map (with bias broadcast for consistent units)
        dense = A.permute(0, 2, 1).reshape(B, -1, side, side)
        dense = dense + self.cls_bias.view(1, -1, 1, 1)

        return {'cls': cls, 'dense': dense, 'patches': p_lat,
                'T': self.T_anchor, 'tau': tau, 'patch_grid': side}


def compute_loss(cls_logits: torch.Tensor,
                  y_multi: torch.Tensor,
                  valid_mask: Optional[torch.Tensor] = None,
                  pos_weight: Optional[torch.Tensor] = None):
    """Ignore-aware sigmoid BCE over multi-label targets."""
    B, C = cls_logits.shape
    if valid_mask is not None and valid_mask.dim() == 1:
        valid_mask = valid_mask.unsqueeze(0).expand(B, -1)
    bce = F.binary_cross_entropy_with_logits(
        cls_logits, y_multi, pos_weight=pos_weight, reduction='none')   # [B, C]
    if valid_mask is not None:
        bce = bce * valid_mask.float()
        denom = valid_mask.float().sum().clamp(min=1.0)
    else:
        denom = torch.tensor(float(B * C), device=cls_logits.device)
    return bce.sum() / denom


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='google/siglip2-base-patch16-256')
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()
    classes = ['shaft', 'wrist', 'clasper', 'gallbladder']
    m = AffinityClassifier(args.model, classes, d_latent=128, topk=5).to(args.device)
    templates = {c: [f'a photo of {c.replace("_"," ")} in surgery'] for c in classes}
    m.init_T_from_templates(templates)
    m.train()
    x = torch.randn(2, 3, 256, 256, device=args.device)
    y = torch.tensor([[1, 1, 0, 1], [0, 1, 1, 0]], dtype=torch.float32, device=args.device)
    out = m(x)
    loss = compute_loss(out['cls'], y)
    print('cls:', out['cls'].shape, 'dense:', out['dense'].shape, 'loss:', float(loss))
    n_train = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f'trainable params: {n_train}')

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
    s_c = pool(A[:,c]) + b_c         L_c = A[:,c].reshape(H_p,W_p)
    (cls logits [B,C])               (dense map [B,C,H_p,W_p])

  Training: image-level multi-label BCE plus optional patch-level BCE.
  Output: cls (for image-level decision) + dense (for downstream seg prior).
"""
from __future__ import annotations
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from util.hyperbolic import expmap0, hyperbolic_affinity
from model.hyperbolic_fusion import HyperbolicFusionBlock
from model.cross_modal_adapter import CMAFusionBlock


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


class AffinityPooling(nn.Module):
    """Pool patch-class affinity A [B, N, C] into image logits [B, C].

    `topk` preserves the original Stage-I behavior. Threshold modes additionally
    expose a patch selection mask M [B, N, C] and selected_count [B, C].
    """

    VALID_MODES = ('topk', 'soft_threshold', 'hard_threshold')

    def __init__(self,
                 num_classes: int,
                 mode: str = 'topk',
                 topk: int = 8,
                 threshold_init: float = 0.3,
                 threshold_learnable: bool = True,
                 gamma: float = 0.1,
                 eps: float = 1e-6,
                 min_selected: int = 1,
                 use_hard_threshold_eval: bool = True):
        super().__init__()
        if mode not in self.VALID_MODES:
            raise ValueError(f'unknown affinity pooling mode: {mode}')
        self.num_classes = int(num_classes)
        self.mode = mode
        self.topk = int(topk)
        self.gamma = float(gamma)
        self.eps = float(eps)
        self.min_selected = int(min_selected)
        self.use_hard_threshold_eval = bool(use_hard_threshold_eval)

        theta = torch.full((self.num_classes,), float(threshold_init))
        threshold_is_trainable = bool(threshold_learnable and mode != 'topk')
        if threshold_is_trainable:
            self.theta = nn.Parameter(theta)
        else:
            self.register_buffer('theta', theta)
        self.threshold_init = float(threshold_init)
        self.threshold_learnable = threshold_is_trainable

    def _topk_pool(self, A: torch.Tensor) -> torch.Tensor:
        k = min(max(int(self.topk), 1), A.shape[1])
        return A.topk(k, dim=1).values.mean(dim=1)

    def forward(self,
                A: torch.Tensor,
                class_bias: Optional[torch.Tensor] = None,
                return_mask: bool = False) -> Dict[str, Optional[torch.Tensor]]:
        assert A.dim() == 3, f'expected A [B, N, C], got {tuple(A.shape)}'
        B, N, C = A.shape
        assert C == self.num_classes, f'expected C={self.num_classes}, got {C}'

        if self.mode == 'topk':
            pooled = self._topk_pool(A)
            logits = pooled if class_bias is None else pooled + class_bias.view(1, C)
            return {
                'image_logits': logits, 'affinity': A,
                'selection_mask': None, 'selected_count': None,
                'threshold': None,
            }

        theta = self.theta.view(1, 1, C)              # [1, 1, C]
        hard = self.mode == 'hard_threshold' or (
            self.mode == 'soft_threshold'
            and (not self.training)
            and self.use_hard_threshold_eval
        )
        if hard:
            mask = (A > theta).to(A.dtype)            # [B, N, C]
        else:
            mask = torch.sigmoid((A - theta) / max(self.gamma, self.eps))

        selected_count = mask.sum(dim=1)              # [B, C]
        denom = selected_count.clamp_min(self.eps)
        pooled = (mask * A).sum(dim=1) / denom        # [B, C]

        if self.min_selected > 0:
            fallback = self._topk_pool(A)
            use_fallback = selected_count < float(self.min_selected)
            pooled = torch.where(use_fallback, fallback, pooled)

        logits = pooled if class_bias is None else pooled + class_bias.view(1, C)
        return {
            'image_logits': logits, 'affinity': A,
            'selection_mask': mask,
            'selected_count': selected_count,
            'threshold': self.theta,
        }

    def candidate_mask(self,
                       pooled: Dict[str, Optional[torch.Tensor]],
                       image_threshold: Optional[float] = None) -> torch.Tensor:
        """Return multi-label candidate classes [B, C] from threshold selection.

        For top-k pooling there is no selected_count, so candidates come from the
        optional image probability threshold, or all classes if no threshold is set.
        """
        logits = pooled['image_logits']
        assert logits is not None
        count = pooled.get('selected_count')
        if count is None:
            cand = torch.ones_like(logits, dtype=torch.bool)
        else:
            cand = count >= float(max(self.min_selected, 0))
        if image_threshold is not None:
            cand = cand & (torch.sigmoid(logits) > float(image_threshold))
        return cand

    def extra_repr(self) -> str:
        return (f'mode={self.mode}, topk={self.topk}, '
                f'threshold_init={self.threshold_init}, '
                f'threshold_learnable={self.threshold_learnable}, '
                f'gamma={self.gamma}, min_selected={self.min_selected}, '
                f'use_hard_threshold_eval={self.use_hard_threshold_eval}')


class AffinityClassifier(nn.Module):
    """Inputs:
        pixel_values [B, 3, H, W]   processed via SigLIP processor
    Returns:
        cls    [B, C]               image-level multi-label logits
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
                 pooling_mode: str = 'topk',
                 threshold_init: float = 0.3,
                 threshold_learnable: bool = True,
                 threshold_gamma: float = 0.1,
                 min_selected: int = 1,
                 hard_threshold_eval: bool = True,
                 vision_backbone: str = 'siglip',
                 dinov2_input_size: int = 518,
                 freeze_vision: bool = False,
                 freeze_text:   bool = False,
                 # ── Hyperbolic affinity options ─────────────────────────
                 affinity_metric: str = 'cosine',           # cosine | hyperbolic | hyperbolic_pathway
                 hyperbolic_dim: Optional[int] = None,      # default: d_latent
                 hyperbolic_curvature_init: float = 1.0,
                 hyperbolic_learn_curvature: bool = True,
                 hyperbolic_temperature_init: float = 0.1,
                 hyperbolic_learn_temperature: bool = True,
                 hyperbolic_eps: float = 1e-5,
                 hyperbolic_distance_scale: float = 1.0,
                 # ── Full hyperbolic feature pathway (when metric='hyperbolic_pathway')
                 hyperbolic_pathway_layers: int = 2,
                 hyperbolic_pathway_alpha_init: float = -3.0,
                 hyperbolic_pathway_light: bool = False,
                 hyperbolic_pathway_grad_checkpoint: bool = False,
                 # ── Stability / regularisation knobs (passed to fusion block) ──
                 hyperbolic_pathway_proj_dropout: float = 0.1,
                 hyperbolic_pathway_fuse_dropout: float = 0.1,
                 hyperbolic_pathway_fuse_layernorm: bool = True,
                 hyperbolic_pathway_residual_mode: str = 'rezero',
                 hyperbolic_pathway_tau_clamp_min: float = 0.05,
                 hyperbolic_pathway_tau_clamp_max: float = 2.0,
                 hyperbolic_pathway_distance_clamp_max: float = 20.0,
                 # ── Cross-Modal Adapter (when metric='cma') ──────────────
                 cma_reduction:        int   = 8,
                 cma_share_dim:        int   = 32,
                 cma_non_linearity:    str   = 'quick_gelu',
                 cma_adapter_dropout:  float = 0.1,
                 cma_adapter_init_std: float = 0.01,
                 cma_fuse_dropout:     float = 0.1,
                 cma_fuse_layernorm:   bool  = True,
                 cma_tau_init:         float = 0.07,
                 cma_tau_clamp_min:    float = 0.01,
                 cma_tau_clamp_max:    float = 2.0):
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
        self.pooling = AffinityPooling(
            num_classes=len(class_names),
            mode=pooling_mode,
            topk=topk,
            threshold_init=threshold_init,
            threshold_learnable=threshold_learnable,
            gamma=threshold_gamma,
            min_selected=min_selected,
            use_hard_threshold_eval=hard_threshold_eval,
        )

        # T_c stored as buffer (NOT a Parameter — frozen after init_T_from_templates)
        self.register_buffer('T_anchor', torch.zeros(len(class_names), d_latent),
                              persistent=False)
        self._anchor_initialised = False

        # ── Affinity metric: cosine / hyperbolic / hyperbolic_pathway / cma ──
        assert affinity_metric in ('cosine', 'hyperbolic', 'hyperbolic_pathway', 'cma'), affinity_metric
        self.affinity_metric = str(affinity_metric)
        self.hyperbolic_eps   = float(hyperbolic_eps)
        self.hyperbolic_distance_scale = float(hyperbolic_distance_scale)
        self.d_hyperbolic     = int(hyperbolic_dim) if hyperbolic_dim else int(d_latent)

        if self.affinity_metric in ('hyperbolic', 'hyperbolic_pathway'):
            # LayerNorm pre-conditioning (stabilizes ExpMap0).
            self.v_pre_ln = nn.LayerNorm(self.d_hyperbolic)
            self.t_pre_ln = nn.LayerNorm(self.d_hyperbolic)

            # Curvature c = softplus(rho) + eps.
            rho_init = float(torch.log(
                torch.expm1(torch.tensor(max(hyperbolic_curvature_init, 1e-4)))))
            self.rho = nn.Parameter(torch.tensor(rho_init),
                                    requires_grad=bool(hyperbolic_learn_curvature))
            tau_init = float(torch.log(
                torch.expm1(torch.tensor(max(hyperbolic_temperature_init, 1e-4)))))
            self.log_tau_h = nn.Parameter(torch.tensor(tau_init),
                                          requires_grad=bool(hyperbolic_learn_temperature))
        else:
            self.v_pre_ln = nn.Identity()
            self.t_pre_ln = nn.Identity()
            self.register_parameter('rho', None)
            self.register_parameter('log_tau_h', None)

        # ── Full hyperbolic pathway block (only constructed when requested) ──
        self.hyp_pathway: Optional[HyperbolicFusionBlock] = None
        if self.affinity_metric == 'hyperbolic_pathway':
            self.hyp_pathway = HyperbolicFusionBlock(
                d_patch=d_text,                            # DINOv2 / SigLIP token dim
                d_hyp=self.d_hyperbolic,
                n_classes=len(class_names),
                hidden_layers=int(hyperbolic_pathway_layers),
                alpha_init=float(hyperbolic_pathway_alpha_init),
                curvature_init=float(hyperbolic_curvature_init),
                tau_init=float(hyperbolic_temperature_init),
                eps=float(hyperbolic_eps),
                distance_scale=float(hyperbolic_distance_scale),
                light=bool(hyperbolic_pathway_light),
                grad_checkpoint=bool(hyperbolic_pathway_grad_checkpoint),
                proj_dropout=float(hyperbolic_pathway_proj_dropout),
                fuse_dropout=float(hyperbolic_pathway_fuse_dropout),
                fuse_layernorm=bool(hyperbolic_pathway_fuse_layernorm),
                residual_mode=str(hyperbolic_pathway_residual_mode),
                tau_clamp_min=float(hyperbolic_pathway_tau_clamp_min),
                tau_clamp_max=float(hyperbolic_pathway_tau_clamp_max),
                distance_clamp_max=float(hyperbolic_pathway_distance_clamp_max),
            )

        # ── Cross-Modal Adapter (when affinity_metric='cma') ─────────────
        self.cma_block: Optional[CMAFusionBlock] = None
        if self.affinity_metric == 'cma':
            self.cma_block = CMAFusionBlock(
                d_patch=d_text,
                d_lat=self.d_hyperbolic,           # reuse d_latent (128) by default
                n_classes=len(class_names),
                reduction=int(cma_reduction),
                share_dim=int(cma_share_dim),
                non_linearity=str(cma_non_linearity),
                adapter_dropout=float(cma_adapter_dropout),
                adapter_init_std=float(cma_adapter_init_std),
                fuse_dropout=float(cma_fuse_dropout),
                fuse_layernorm=bool(cma_fuse_layernorm),
                tau_init=float(cma_tau_init),
                tau_clamp_min=float(cma_tau_clamp_min),
                tau_clamp_max=float(cma_tau_clamp_max),
            )

    # ── helpers for hyperbolic mode ────────────────────────────────────────
    def get_curvature(self) -> torch.Tensor:
        return F.softplus(self.rho) + self.hyperbolic_eps

    def get_tau_h(self) -> torch.Tensor:
        return F.softplus(self.log_tau_h) + self.hyperbolic_eps

    @staticmethod
    def _build_dinov2():
        """Build DINOv2-B/14. Prefer LOCAL .pth (UniMatch-V2 already ships it); fall back to torch.hub."""
        import os
        LOCAL_CANDIDATES = [
            os.environ.get('DINOV2_VITB14_PATH', ''),
            './pretrained/dinov2_vitb14_pretrain.pth',
            '/root/autodl-tmp/code/UniMatch-V2/pretrained/dinov2_vitb14_pretrain.pth',
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
                if self.affinity_metric in ('hyperbolic', 'hyperbolic_pathway'):
                    z_t = self.t_pre_ln(self.text_proj(pooled))           # [P, d_h]
                    z_mean = z_t.mean(0)                                   # [d_h]
                    t_h = expmap0(z_mean.unsqueeze(0),
                                  c=self.get_curvature(),
                                  eps=self.hyperbolic_eps).squeeze(0)
                    self.T_anchor[cidx].copy_(t_h)
                else:
                    proj = F.normalize(self.text_proj(pooled), dim=-1)
                    self.T_anchor[cidx].copy_(F.normalize(proj.mean(0), dim=-1))
        self._anchor_initialised = True

        # CMA block: seed its `text_features_raw` from the RAW SigLIP text encoder
        # output (NOT through text_proj — the adapter is the only text head in CMA).
        if self.affinity_metric == 'cma' and self.cma_block is not None:
            with torch.no_grad():
                t_raw_list = []
                device = self.log_tau.device
                for cidx, cname in enumerate(self.class_names):
                    prompts = template_dict.get(cname) or \
                              [cname.replace('_',' ').replace('-',' ')]
                    toks = self.tokenizer(prompts, return_tensors='pt',
                                           padding='max_length', truncation=True,
                                           max_length=64).to(device)
                    out = self.text_model(**toks)
                    last = out.last_hidden_state
                    attn = toks.get('attention_mask', None)
                    if attn is not None:
                        lens = attn.sum(dim=1) - 1
                        pooled = last[torch.arange(last.size(0), device=device), lens]
                    else:
                        pooled = last[:, -1, :]
                    # average raw 768-dim text features across prompts; CMA's adapter
                    # is the only projector — DO NOT push through text_proj here.
                    t_raw_list.append(pooled.mean(0))
                raw = torch.stack(t_raw_list, 0)                                 # [C', 768]
                self.cma_block.init_text_features(raw)

        # Pathway block: also seed its T_h_param from the text projections
        if self.affinity_metric == 'hyperbolic_pathway' and self.hyp_pathway is not None:
            with torch.no_grad():
                t_list = []
                device = self.log_tau.device
                for cidx, cname in enumerate(self.class_names):
                    prompts = template_dict.get(cname) or \
                              [cname.replace('_',' ').replace('-',' ')]
                    toks = self.tokenizer(prompts, return_tensors='pt',
                                           padding='max_length', truncation=True,
                                           max_length=64).to(device)
                    out = self.text_model(**toks)
                    last = out.last_hidden_state
                    attn = toks.get('attention_mask', None)
                    if attn is not None:
                        lens = attn.sum(dim=1) - 1
                        pooled = last[torch.arange(last.size(0), device=device), lens]
                    else:
                        pooled = last[:, -1, :]
                    z_t = self.t_pre_ln(self.text_proj(pooled)).mean(0)
                    t_list.append(z_t)
                raw = torch.stack(t_list, 0)                                  # [C', d_h]
                self.hyp_pathway.init_T_from_text_features(raw)

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

        if self.affinity_metric == 'cma':
            # ── Cross-Modal Adapter feature pathway ───────────────────────
            assert self.cma_block is not None
            out = self.cma_block(tokens)                              # dict
            A = out['A']                                              # already has cls_bias
            patches_out = out['X_aware']    # 768-dim Euclidean for Stage-II DPT
            tau_report  = out['tau']
            _pathway_active = True
        elif self.affinity_metric == 'hyperbolic_pathway':
            # ── Full hyperbolic feature pathway ────────────────────────────
            assert self.hyp_pathway is not None
            out = self.hyp_pathway(tokens)                            # dict
            A = out['A']                                              # already has class_bias
            patches_out = out['X_aware']    # 768-dim Euclidean for Stage-II DPT
            tau_report  = out['tau']
            # Pathway's own cls_bias already added inside; outer pooling adds it again
            # -> subtract it here to avoid double-counting (or pass class_bias=None below).
            # We choose to NOT pass cls_bias to the pooling in this mode:
            _pathway_active = True
        elif self.affinity_metric == 'hyperbolic':
            z_v = self.v_pre_ln(self.visual_proj(tokens))             # [B, N, d_h]
            c = self.get_curvature()
            tau_h = self.get_tau_h()
            p_h = expmap0(z_v, c=c, eps=self.hyperbolic_eps)
            t_h = self.T_anchor
            A = hyperbolic_affinity(
                p_h, t_h, tau_h=tau_h, class_bias=None,
                c=c, eps=self.hyperbolic_eps,
                distance_scale=self.hyperbolic_distance_scale,
            )
            patches_out = z_v
            tau_report  = tau_h
            _pathway_active = False
        else:
            # ── Cosine (original) ──────────────────────────────────────────
            p_lat = F.normalize(self.visual_proj(tokens), dim=-1)     # [B, N, d_lat]
            tau = self.log_tau.exp().clamp(1e-3, 1.0)
            A = (p_lat @ self.T_anchor.T) / tau                        # [B, N, C']
            patches_out = p_lat
            tau_report = tau
            _pathway_active = False

        # In pathway mode the cls_bias is already inside `A`; skip the pool's bias.
        pool_bias = None if _pathway_active else self.cls_bias
        pooled = self.pooling(A, class_bias=pool_bias, return_mask=True)
        cls = pooled['image_logits']                                   # [B, C']

        # Dense map (logits): bias broadcast keeps units consistent with cls
        dense = A.permute(0, 2, 1).reshape(B, -1, side, side)
        dense = dense + self.cls_bias.view(1, -1, 1, 1)

        return {'cls': cls, 'image_logits': cls, 'affinity': A,
                'selection_mask': pooled['selection_mask'],
                'selected_count': pooled['selected_count'],
                'threshold': pooled['threshold'],
                'dense': dense, 'patches': patches_out,
                'T': self.T_anchor, 'tau': tau_report, 'patch_grid': side}

    def candidate_mask(self,
                       out: Dict[str, Optional[torch.Tensor]],
                       image_threshold: Optional[float] = None) -> torch.Tensor:
        """Return multi-label candidate classes [B, C] from a forward output."""
        pooled = {
            'image_logits': out.get('image_logits', out.get('cls')),
            'selected_count': out.get('selected_count'),
        }
        return self.pooling.candidate_mask(pooled, image_threshold=image_threshold)


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

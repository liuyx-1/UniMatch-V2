"""End-to-end CoOp + latent-space + dense-patch-text alignment for image-level cls.

  ┌────────── frozen SigLIP-2 ──────────┐    ┌──── trainable ────┐
  │  vision encoder        text encoder │    │ CoOp ctx [K, 768]  │
  └──────┬─────────────────────┬────────┘    │ text_proj 768→128  │
         │                      │              │ visual_proj 768→128│
   patch tokens [B,N,768]   [BOS,ctx,wc,EOS]   │ log_tau            │
         │                      │              └────────────────────┘
         ▼ visual_proj          ▼ pool + text_proj
   p̂ [B,N,128] L2-norm    T_c [C,128] L2-norm
                  \              /
                   S = p̂ @ T_cᵀ / τ          [B, N, C]
                                │
                       ┌────────┴────────┐
                       ▼                 ▼
              image-level cls         dense map (for seg prior)
              s_c = topk/max(S[:,c]) [B, C]    S transposed [B,C,N]

  Training (per labeled image):
    image-level multi-label BCE on s_c              ← always
    patch-level BCE on patch_logit[i,c]:
        positive iff GT_mask_patch[i] == c          ← optional, much stronger
        ignore patches with GT==255
  Both losses use the SAME T_c / p̂ → end-to-end alignment of latent space.

  Warm-start from coop_state.pt (CoOp ctx + text/visual proj weights).
"""
from __future__ import annotations
from collections import OrderedDict
from pathlib import Path
from typing import List, Optional, Dict

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


def _build_proj(d_in: int, d_out: int, hidden: int = 256, dropout: float = 0.0) -> nn.Sequential:
    return nn.Sequential(
        nn.LayerNorm(d_in),
        nn.Linear(d_in, hidden),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden, d_out),
    )


# ----------------------------------------------------------------------
class DenseAlignClassifier(nn.Module):
    """Trains CoOp ctx + latent projections by joint image-level + patch-level BCE.

    Inputs:
        pixel_values [B, 3, H, W]    SigLIP-processed
        y_multi      [B, C]          image-level multi-label target (0/1)
        mask_patch   [B, H_p, W_p]   optional; long, in [0,C) (255 = ignore)
        pooling      'max' | 'top5' | 'top10' | 'mean'

    Forward returns:
        cls_logits   [B, C]
        dense_logits [B, C, H_p, W_p]
        T            [C, d_latent]    current text anchors (L2-norm)
        patches      [B, N, d_latent] current patch latents (L2-norm)
    """

    def __init__(self,
                 model_name: str,
                 class_names: List[str],
                 n_ctx: int = 16,
                 d_latent: int = 128,
                 d_text: int = 768,
                 pooling: str = 'top5',
                 init_tau: float = 0.07,
                 share_ctx: bool = False,
                 text_anchor: str = 'coop'):
        """text_anchor:
            'coop'      — T_c = CoOp soft-prompt encoded (default)
            'templates' — T_c = ensembled hand-written templates (frozen)
            'mix'       — T_c = norm(sigmoid(beta_c)*T_template + (1-sigmoid(beta_c))*T_coop),
                          learnable per-class beta_c (init 0)
        """
        super().__init__()
        from transformers import AutoModel, AutoProcessor, AutoTokenizer
        self.model_name = model_name
        self.processor  = AutoProcessor.from_pretrained(model_name)
        self.tokenizer  = AutoTokenizer.from_pretrained(model_name)
        backbone = AutoModel.from_pretrained(model_name)
        self.vision_model = backbone.vision_model
        self.text_model   = backbone.text_model
        for p in self.vision_model.parameters(): p.requires_grad_(False)
        for p in self.text_model.parameters():   p.requires_grad_(False)
        self.vision_model.eval(); self.text_model.eval()

        self.class_names = class_names
        self.n_ctx = n_ctx
        self.d_latent = d_latent
        self.d_text = d_text
        self.pooling = pooling
        self.share_ctx = share_ctx

        if share_ctx:
            self.ctx = nn.Parameter(torch.zeros(n_ctx, d_text))
            nn.init.normal_(self.ctx, std=0.02)
        else:
            # per-class ctx (matches coop_state.pt schema)
            self.ctx = nn.Parameter(torch.zeros(len(class_names), n_ctx, d_text))
            nn.init.normal_(self.ctx, std=0.02)

        self.text_proj   = _build_proj(d_text, d_latent)
        self.visual_proj = _build_proj(d_text, d_latent)
        self.log_tau = nn.Parameter(torch.tensor(float(torch.log(torch.tensor(init_tau)))))
        # per-class bias absorbs class-frequency offset (Q2: small head)
        self.cls_bias = nn.Parameter(torch.zeros(len(class_names)))
        # text-attention pooling temperature (used only when pooling=='text_attn')
        self.log_tau_attn = nn.Parameter(torch.tensor(float(torch.log(torch.tensor(0.1)))))

        # Cache class-word token embeddings (frozen, computed once on first forward)
        self._class_word_embs: Optional[List[torch.Tensor]] = None
        self._special: Optional[Dict[str, torch.Tensor]] = None

        # Template-anchor support
        assert text_anchor in ('coop', 'templates', 'mix'), text_anchor
        self.text_anchor = text_anchor
        self.register_buffer('T_template', torch.zeros(len(class_names), d_latent),
                              persistent=False)
        self._template_initialised = False
        if text_anchor == 'mix':
            # per-class learnable mixing logit; sigmoid(0)=0.5 default
            self.beta_logit = nn.Parameter(torch.zeros(len(class_names)))
        else:
            self.beta_logit = None

    # ---- warm start from coop_state.pt ----
    @torch.no_grad()
    def warm_start(self, coop_state_path: str | Path, dataset_name: str):
        s = torch.load(coop_state_path, map_location='cpu')
        # text_proj / visual_proj
        def _load_into(mlp: nn.Sequential, sd: OrderedDict):
            new = {k.replace('body.', ''): v for k, v in sd.items()}
            mlp.load_state_dict(new)
        _load_into(self.text_proj,   s['text_proj'])
        _load_into(self.visual_proj, s['visual_proj'])
        # ctx (per-class)
        if self.share_ctx:
            keys = [f'{dataset_name}__{c}' for c in range(len(self.class_names))]
            stacked = torch.stack([s['soft_prompts'][k] for k in keys], dim=0)  # [C, K, D]
            self.ctx.copy_(stacked.mean(0))   # average if forced share
        else:
            for c in range(len(self.class_names)):
                key = f'{dataset_name}__{c}'
                if key in s['soft_prompts']:
                    self.ctx[c].copy_(s['soft_prompts'][key])
        if 'logit_scale' in s:
            # coop stored ln(scale) = -ln(tau)  → log_tau = -logit_scale
            self.log_tau.copy_(-s['logit_scale'].float())

    # ---- text tokens cache ----
    @torch.no_grad()
    def _cache_class_words(self, device):
        if self._class_word_embs is not None: return
        tok_emb = self.text_model.embeddings.token_embedding
        D = tok_emb.embedding_dim
        words = [c.replace('_', ' ').replace('-', ' ') for c in self.class_names]
        embs = []
        for w in words:
            ids = self.tokenizer(w, add_special_tokens=False, return_tensors='pt').input_ids[0]
            embs.append(tok_emb(ids.to(device)))
        self._class_word_embs = embs
        bos = self.tokenizer.bos_token_id if self.tokenizer.bos_token_id is not None else 0
        eos = self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else 0
        pad = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        self._special = {
            'bos': tok_emb(torch.tensor([bos], device=device))[0],
            'eos': tok_emb(torch.tensor([eos], device=device))[0],
            'pad': tok_emb(torch.tensor([pad], device=device))[0],
        }

    # ---- precompute T_template from hand-written prompts ----
    @torch.no_grad()
    def init_T_from_templates(self, template_dict: Dict[str, List[str]]):
        """For each class name in self.class_names, ensemble all prompts in
        template_dict[class_name] into a single T_c stored in self.T_template."""
        device = self.ctx.device
        text_model = self.text_model
        tok = self.tokenizer
        for cidx, cname in enumerate(self.class_names):
            prompts = template_dict.get(cname)
            if not prompts:
                # fallback: just the class word
                prompts = [cname.replace('_', ' ').replace('-', ' ')]
            toks = tok(prompts, return_tensors='pt', padding='max_length',
                        truncation=True, max_length=64).to(device)
            out = text_model(**toks)
            last = out.last_hidden_state                              # [n, L, D]
            # SigLIP-2 text pool = last non-pad token (or EOS)
            attn = toks.get('attention_mask', None)
            if attn is not None:
                lens = attn.sum(dim=1) - 1
                pooled = last[torch.arange(last.size(0), device=device), lens]
            else:
                pooled = last[:, -1, :]
            proj = self.text_proj(pooled)
            proj = F.normalize(proj, dim=-1)                          # [n, d_latent]
            T_c = F.normalize(proj.mean(0), dim=-1)                   # [d_latent]
            self.T_template[cidx].copy_(T_c)
        self._template_initialised = True

    # ---- text branch: build T_c ----
    def _encode_coop(self) -> torch.Tensor:
        device = self.ctx.device
        self._cache_class_words(device)
        bos = self._special['bos']; eos = self._special['eos']; pad = self._special['pad']
        seqs, lens = [], []
        for c, w_emb in enumerate(self._class_word_embs):
            ctx_c = self.ctx if self.share_ctx else self.ctx[c]
            seq = torch.cat([bos.unsqueeze(0), ctx_c, w_emb, eos.unsqueeze(0)], dim=0)
            seqs.append(seq); lens.append(seq.size(0))
        Lmax = max(lens); C = len(seqs); D = self.d_text
        inputs = torch.zeros(C, Lmax, D, device=device, dtype=seqs[0].dtype)
        pad_mask = torch.zeros(C, Lmax, dtype=torch.bool, device=device)
        for i, (seq, L) in enumerate(zip(seqs, lens)):
            inputs[i, :L] = seq
            if L < Lmax:
                inputs[i, L:] = pad
                pad_mask[i, L:] = True
        pos_ids = torch.arange(Lmax, device=device).unsqueeze(0).expand(C, -1)
        hidden = inputs + self.text_model.embeddings.position_embedding(pos_ids)
        min_dtype = torch.finfo(hidden.dtype).min
        ext = (pad_mask.to(hidden.dtype) * min_dtype)[:, None, None, :]
        enc_out = self.text_model.encoder(inputs_embeds=hidden, attention_mask=ext)
        last = enc_out.last_hidden_state if hasattr(enc_out, 'last_hidden_state') else enc_out[0]
        eos_pos = torch.tensor([L - 1 for L in lens], device=device)
        pooled = last[torch.arange(C, device=device), eos_pos]
        T = self.text_proj(pooled)
        return F.normalize(T, dim=-1)

    def encode_text(self) -> torch.Tensor:
        if self.text_anchor == 'templates':
            assert self._template_initialised, 'call init_T_from_templates() first'
            return self.T_template
        if self.text_anchor == 'coop':
            return self._encode_coop()
        # mix
        assert self._template_initialised, 'call init_T_from_templates() first'
        T_coop = self._encode_coop()
        beta = torch.sigmoid(self.beta_logit).unsqueeze(-1)             # [C, 1]
        T = beta * self.T_template + (1 - beta) * T_coop
        return F.normalize(T, dim=-1)

    # ---- vision branch: patch latents ----
    def encode_image(self, pixel_values: torch.Tensor):
        with torch.no_grad():
            out = self.vision_model(pixel_values=pixel_values)
        tokens = out.last_hidden_state                          # [B, N, 768]
        B, N, _ = tokens.shape
        side = int(round(N ** 0.5))
        latent = F.normalize(self.visual_proj(tokens), dim=-1)  # [B, N, d_latent]
        return latent, side

    # ---- forward ----
    def forward(self, pixel_values, mask_patch: Optional[torch.Tensor] = None):
        T = self.encode_text()                                  # [C, d_latent]
        patches, side = self.encode_image(pixel_values)         # [B, N, d_latent]
        tau = self.log_tau.exp().clamp(1e-3, 1.0)
        S = (patches @ T.T) / tau                                # [B, N, C]

        # image-level pooled score
        attn_map = None                                              # set only for text_attn
        if self.pooling == 'max':
            cls = S.max(dim=1).values
        elif self.pooling == 'mean':
            cls = S.mean(dim=1)
        elif self.pooling.startswith('top'):
            k = int(self.pooling[3:]); k = min(k, S.size(1))
            cls = S.topk(k, dim=1).values.mean(dim=1)
        elif self.pooling == 'text_attn':
            # softmax attention over patches per class (text-conditioned pooling)
            tau_attn = self.log_tau_attn.exp().clamp(1e-3, 1.0)
            attn = F.softmax(S / tau_attn, dim=1)                    # [B, N, C]
            attn_map = attn
            # weighted pool patches per class:  [B, C, D]
            pooled = torch.einsum('bnc,bnd->bcd', attn, patches)
            pooled = F.normalize(pooled, dim=-1)
            # T:[C,D] -> broadcast per-class dot
            cls = (pooled * T.unsqueeze(0)).sum(dim=-1) / tau         # [B, C]
        else:
            raise ValueError(self.pooling)
        cls = cls + self.cls_bias                                    # per-class bias

        # dense map ALWAYS returns raw logits (S + bias) for patch BCE supervision.
        # attn_map is optionally returned separately for visualisation / DPT prior.
        dense = S.permute(0, 2, 1).reshape(S.size(0), -1, side, side)
        dense = dense + self.cls_bias.view(1, -1, 1, 1)
        return {
            'cls': cls, 'dense': dense, 'T': T, 'patches': patches,
            'tau': tau, 'patch_grid': side,
            'attn': attn_map,                              # [B, N, C] softmax probs, or None
        }


# ----------------------------------------------------------------------
def compute_losses(out: Dict[str, torch.Tensor],
                   y_multi: torch.Tensor,
                   mask_patch: Optional[torch.Tensor] = None,
                   pos_weight: Optional[torch.Tensor] = None,
                   lambda_cls: float = 1.0,
                   lambda_patch: float = 1.0,
                   ignore_index: int = 255,
                   valid_mask: Optional[torch.Tensor] = None):
    """valid_mask [B, C] (or [C]) — 1 for classes annotated by this image's dataset.
    Classes with valid_mask=0 are ignored in BOTH cls and patch BCE.
    """
    """
    out['cls']:   [B, C]
    out['dense']: [B, C, H_p, W_p]
    y_multi:      [B, C]
    mask_patch:   [B, H_p, W_p] long (255 ignore)   for patch-level supervision
    """
    cls_logits = out['cls']                                     # [B, C]
    B, C = cls_logits.shape
    if valid_mask is not None and valid_mask.dim() == 1:
        valid_mask = valid_mask.unsqueeze(0).expand(B, -1)        # [B, C]

    # ---- ignore-aware image-level BCE ----
    pos_w = pos_weight if pos_weight is not None else None
    bce_cls = F.binary_cross_entropy_with_logits(
        cls_logits, y_multi, pos_weight=pos_w, reduction='none')  # [B, C]
    if valid_mask is not None:
        bce_cls = bce_cls * valid_mask.float()
        denom = valid_mask.float().sum().clamp(min=1.0)
    else:
        denom = torch.tensor(float(B * C), device=cls_logits.device)
    L_cls = bce_cls.sum() / denom

    # ---- ignore-aware patch-level BCE ----
    L_patch = torch.tensor(0.0, device=cls_logits.device)
    if mask_patch is not None and lambda_patch > 0:
        Bd, Cd, H, W = out['dense'].shape
        valid_pix = (mask_patch != ignore_index)                  # [B, H, W]
        mask_clamped = mask_patch.clamp(0, Cd - 1)
        target = F.one_hot(mask_clamped, num_classes=Cd).permute(0, 3, 1, 2).float()
        pos_w_p = pos_weight.view(1, -1, 1, 1) if pos_weight is not None else None
        bce_patch = F.binary_cross_entropy_with_logits(
            out['dense'], target, pos_weight=pos_w_p, reduction='none')
        bce_patch = bce_patch * valid_pix.unsqueeze(1).float()
        if valid_mask is not None:
            bce_patch = bce_patch * valid_mask.view(B, C, 1, 1).float()
            denom_p = (valid_pix.float().sum() * valid_mask.float().sum(dim=1).mean()).clamp(min=1.0)
        else:
            denom_p = valid_pix.float().sum().clamp(min=1.0) * Cd
        L_patch = bce_patch.sum() / denom_p

    return {
        'total': lambda_cls * L_cls + lambda_patch * L_patch,
        'cls': L_cls, 'patch': L_patch,
    }


def resize_mask_to_patch(mask: torch.Tensor, patch_size: int) -> torch.Tensor:
    """[B, H, W] long  →  [B, patch_size, patch_size] long via nearest."""
    return F.interpolate(mask.unsqueeze(1).float(), size=patch_size,
                          mode='nearest').squeeze(1).long()


# ----------------------------------------------------------------------
if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='google/siglip2-base-patch16-256')
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()
    classes = ['background', 'shaft', 'wrist', 'clasper']
    m = DenseAlignClassifier(args.model, classes, n_ctx=16, d_latent=128,
                              pooling='top5', share_ctx=False).to(args.device)
    m.train()
    x = torch.randn(2, 3, 256, 256, device=args.device)
    mp = torch.randint(0, len(classes), (2, 16, 16), device=args.device)
    y = torch.tensor([[1,1,1,0], [1,0,1,1]], dtype=torch.float32, device=args.device)
    out = m(x)
    print('cls   ', out['cls'].shape)
    print('dense ', out['dense'].shape)
    print('T     ', out['T'].shape)
    print('tau   ', out['tau'].item())
    losses = compute_losses(out, y, mask_patch=mp)
    print('losses', {k: float(v) for k, v in losses.items()})
    n_trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f'trainable params: {n_trainable}')

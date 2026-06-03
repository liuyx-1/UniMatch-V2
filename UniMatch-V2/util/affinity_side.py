"""Patch-text side branch used by the segmentation trainer.

Single visual encoder design: this module does NOT hold its own DINOv2.
It takes PRE-COMPUTED patch tokens [B, N, 768] from the DPT main backbone
and produces:
   - prior_full [B, n_orig, H, W]    dense per-class affinity (cos sim with T_anchor)
   - gate       [B, 1,      H, W]    per-pixel mixing weight, sigmoid output
   - cls_logits [B, C_new]            image-level pooled affinity score

Trainable params:
   - gate_conv             always trainable from step 0
   - visual_proj           default FROZEN when loaded from a warmstart,
                              unfrozen immediately by --joint-text-stage
   - T_anchor, cls_bias, log_tau   same lifecycle as visual_proj

When no warmstart checkpoint is provided, T_anchor is initialized directly from
dataset class names and surgical prompt templates, then learned inside the main
segmentation run.

Fusion: L_final = (1 - g·mask) * L_DPT + g·mask * L_prior
   mask: 1 for non-bg classes (in original DPT class order), 0 for bg.
"""
from __future__ import annotations
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.affinity_cls import AffinityPooling
from util.hyperbolic import expmap0, hyperbolic_affinity
from model.hyperbolic_fusion import HyperbolicFusionBlock
from model.cross_modal_adapter import CMAFusionBlock
from util.edge_enhance import EdgeTextResidualAdapter, edge_to_tokens


def _build_proj(d_in: int, d_out: int, hidden: int = 256, dropout: float = 0.0):
    return nn.Sequential(
        nn.LayerNorm(d_in),
        nn.Linear(d_in, hidden),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden, d_out),
    )


def _normalize_name(name: str) -> str:
    return name.lower().replace(' ', '_').replace('-', '_')


def _templates_for_class(name: str):
    try:
        from tools.class_templates import TEMPLATES
        key = _normalize_name(name)
        if key in TEMPLATES:
            return TEMPLATES[key]
    except Exception:
        pass
    return [f'a photo of {name.replace("_", " ").replace("-", " ")} in surgery']


def _init_text_anchors(class_names, d_latent: int, model_name: str):
    """Initialize class anchors from SigLIP text prompts for joint one-stage runs."""
    from transformers import AutoModel, AutoTokenizer

    text_model = AutoModel.from_pretrained(model_name).text_model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    text_proj = _build_proj(768, int(d_latent)).eval()
    anchors = []
    with torch.no_grad():
        for cname in class_names:
            prompts = _templates_for_class(cname)
            toks = tokenizer(prompts, return_tensors='pt', padding='max_length',
                             truncation=True, max_length=64)
            out = text_model(**toks)
            last = out.last_hidden_state
            attn = toks.get('attention_mask', None)
            if attn is not None:
                lens = attn.sum(dim=1) - 1
                pooled = last[torch.arange(last.size(0)), lens]
            else:
                pooled = last[:, -1, :]
            proj = F.normalize(text_proj(pooled), dim=-1)
            anchors.append(F.normalize(proj.mean(0), dim=-1))
    return torch.stack(anchors, dim=0), text_proj.state_dict()


def _make_joint_init_ckpt(class_names, n_orig_classes: int, dataset: str = 'unknown',
                          model_name: str = 'google/siglip2-base-patch16-256',
                          d_latent: int = 128):
    if class_names is None:
        raise ValueError('class_names is required when initializing affinity side branch without a warmstart checkpoint')
    keep_orig_ids = [i for i, name in enumerate(class_names)
                     if not str(name).lower().startswith('background')]
    new_names = [class_names[i] for i in keep_orig_ids]
    if not new_names:
        raise ValueError('no non-background classes available for affinity side branch initialization')
    T_anchor, visual_state = _init_text_anchors(new_names, d_latent=d_latent,
                                                model_name=model_name)
    return {
        'dataset': dataset,
        'd_latent': int(d_latent),
        'topk': 5,
        'pooling_mode': 'soft_threshold',
        'threshold_init': 0.3,
        'threshold_learnable': True,
        'threshold_gamma': 0.1,
        'min_selected': 1,
        'hard_threshold_eval': True,
        'affinity_metric': 'cosine',
        'visual_proj': visual_state,
        'T_anchor': T_anchor,
        'cls_bias': torch.zeros(len(new_names)),
        'log_tau': torch.tensor(float(torch.log(torch.tensor(0.07)))),
        'orig_to_new': {int(orig): int(new) for new, orig in enumerate(keep_orig_ids)},
        'class_names': new_names,
        'orig_class_names': list(class_names),
        'joint_initialized': True,
    }


class AffinitySideBranch(nn.Module):
    """Operates on patch tokens produced elsewhere; no DINOv2 inside."""

    def __init__(self, ckpt_path: str = None, n_orig_classes: int = None,
                 gate_bias_init: float = -3.0, class_names=None,
                 dataset: str = 'unknown',
                 model_name: str = 'google/siglip2-base-patch16-256'):
        super().__init__()
        if isinstance(ckpt_path, dict):
            ckpt = ckpt_path
            self.ckpt_path = '<checkpoint-affinity-side>'
        elif ckpt_path:
            ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            self.ckpt_path = str(ckpt_path)
        else:
            ckpt = _make_joint_init_ckpt(class_names, int(n_orig_classes),
                                         dataset=dataset, model_name=model_name)
            self.ckpt_path = '<joint-init-from-class-names>'
        self.dataset = ckpt.get('dataset', 'unknown')

        d_text   = 768
        d_latent = int(ckpt.get('d_latent', 128))
        self.d_text   = d_text
        self.d_latent = d_latent
        self.topk     = int(ckpt.get('topk', 5))
        self.pooling_mode = ckpt.get('pooling_mode', 'topk')

        # ── Affinity metric: 'cosine' (legacy) or 'hyperbolic' ─────────────
        self.affinity_metric = str(ckpt.get('affinity_metric', 'cosine'))
        self.hyperbolic_eps  = float(ckpt.get('hyperbolic_eps', 1e-5))
        self.hyperbolic_distance_scale = float(ckpt.get('hyperbolic_distance_scale', 1.0))
        self.d_hyperbolic    = int(ckpt.get('hyperbolic_dim', d_latent))

        # visual_proj — loaded from warmstart/checkpoint or initialized from text prompts.
        self.visual_proj = _build_proj(d_text, d_latent)
        self.visual_proj.load_state_dict(ckpt['visual_proj'])

        # Hyperbolic-specific parameters (only used when metric='hyperbolic')
        if self.affinity_metric == 'hyperbolic':
            self.v_pre_ln = nn.LayerNorm(self.d_hyperbolic)
            if ckpt.get('hyperbolic_v_pre_ln') is not None:
                self.v_pre_ln.load_state_dict(ckpt['hyperbolic_v_pre_ln'])
            self.rho       = nn.Parameter(ckpt['hyperbolic_rho'].float().clone(),       requires_grad=False)
            self.log_tau_h = nn.Parameter(ckpt['hyperbolic_log_tau_h'].float().clone(), requires_grad=False)
        else:
            self.v_pre_ln = nn.Identity()
            self.register_parameter('rho', None)
            self.register_parameter('log_tau_h', None)

        # ── Full hyperbolic feature pathway (loaded if the warmstart/checkpoint used one) ─
        self.hyp_pathway = None
        if self.affinity_metric == 'hyperbolic_pathway':
            pst = ckpt.get('hyp_pathway_state')
            assert pst is not None, "ckpt declares hyperbolic_pathway but is missing 'hyp_pathway_state'"
            self.hyp_pathway = HyperbolicFusionBlock(
                d_patch=int(pst.get('d_patch', d_text)),
                d_hyp=int(pst.get('d_hyp', d_latent)),
                n_classes=int(pst.get('n_classes', int(ckpt['T_anchor'].shape[0]))),
                hidden_layers=int(pst.get('hidden_layers',
                                          ckpt.get('hyp_pathway_layers', 2))),
                alpha_init=float(ckpt.get('hyp_pathway_alpha_init', -3.0)),
                curvature_init=1.0, tau_init=0.1,
                eps=float(pst.get('eps', 1e-5)),
                distance_scale=float(pst.get('distance_scale', 1.0)),
                light=bool(pst.get('light', False)),
                grad_checkpoint=False,    # inference/training-side default off
                # ── Mirror warmstart config so weights load cleanly ──
                proj_dropout=float(pst.get('proj_dropout', 0.0)),     # off at inference-style
                fuse_dropout=float(pst.get('fuse_dropout_p', 0.0)),
                fuse_layernorm=bool(pst.get('fuse_layernorm', False)),
                residual_mode=str(pst.get('residual_mode', 'sigmoid_gate')),
                tau_clamp_min=float(pst.get('tau_clamp_min', 1e-5)),
                tau_clamp_max=float(pst.get('tau_clamp_max', 1e6)),
                distance_clamp_max=float(pst.get('distance_clamp_max', 1e6)),
            )
            self.hyp_pathway.load_state_dict_from_save(pst)

        # ── Cross-Modal Adapter fusion block (when metric='cma') ────────
        self.cma_block = None
        if self.affinity_metric == 'cma':
            cst = ckpt.get('cma_block_state')
            assert cst is not None, "ckpt declares metric='cma' but is missing 'cma_block_state'"
            self.cma_block = CMAFusionBlock(
                d_patch=int(cst.get('d_patch', d_text)),
                d_lat=int(cst.get('d_lat', d_latent)),
                n_classes=int(cst.get('n_classes', int(ckpt['T_anchor'].shape[0]))),
                reduction=int(cst.get('r', ckpt.get('cma_reduction', 8))),
                share_dim=int(cst.get('d_s', ckpt.get('cma_share_dim', 32))),
                non_linearity=str(cst.get('non_linearity', ckpt.get('cma_non_linearity', 'quick_gelu'))),
                adapter_dropout=0.0,                # default off at inference/training-side use
                adapter_init_std=0.01,
                fuse_dropout=float(cst.get('fuse_dropout_p', 0.0)),
                fuse_layernorm=bool(cst.get('fuse_layernorm', True)),
                tau_init=0.07,
                tau_clamp_min=float(cst.get('tau_clamp_min', 0.01)),
                tau_clamp_max=float(cst.get('tau_clamp_max', 2.0)),
            )
            self.cma_block.load_state_dict_from_save(cst)

        # T_anchor: stored directly. Learnable param so it can be unfrozen.
        T_anchor = ckpt['T_anchor'].float()                      # [C_new, d_latent]
        self.T_anchor = nn.Parameter(T_anchor.clone(), requires_grad=False)
        self.cls_bias = nn.Parameter(ckpt['cls_bias'].float().clone(), requires_grad=False)
        self.log_tau  = nn.Parameter(ckpt['log_tau'].float().clone(), requires_grad=False)
        self.pooling = AffinityPooling(
            num_classes=int(T_anchor.shape[0]),
            mode=self.pooling_mode,
            topk=self.topk,
            threshold_init=float(ckpt.get('threshold_init', 0.3)),
            threshold_learnable=bool(ckpt.get('threshold_learnable', False)),
            gamma=float(ckpt.get('threshold_gamma', 0.1)),
            min_selected=int(ckpt.get('min_selected', 1)),
            use_hard_threshold_eval=bool(ckpt.get('hard_threshold_eval', True)),
        )
        if 'threshold' in ckpt and ckpt['threshold'] is not None:
            self.pooling.theta.data.copy_(ckpt['threshold'].float())

        # Class index mappings (orig DPT-id space → new affinity-id space, -1 for bg)
        self.n_orig = int(n_orig_classes)
        if 'orig_to_new' in ckpt:
            orig_to_new = {int(k): int(v) for k, v in ckpt['orig_to_new'].items()}
            idx = [orig_to_new.get(c, -1) for c in range(self.n_orig)]
        elif 'idx_orig_to_new' in ckpt:
            idx = [int(v) for v in ckpt['idx_orig_to_new']]
            if len(idx) != self.n_orig:
                raise ValueError(f'idx_orig_to_new length {len(idx)} does not match n_orig_classes={self.n_orig}')
        else:
            raise KeyError("affinity checkpoint/state is missing 'orig_to_new' or 'idx_orig_to_new'")
        self.register_buffer('idx_orig_to_new',
                              torch.tensor(idx, dtype=torch.long))
        self.register_buffer('non_bg_mask',
                              (self.idx_orig_to_new >= 0).to(torch.float32).view(1, -1, 1, 1))

        # gate_conv: 1×1 on the d_latent feature, init near 0 (bias=-3 → σ≈0.05)
        self.gate_conv = nn.Conv2d(d_latent, 1, kernel_size=1)
        nn.init.zeros_(self.gate_conv.weight)
        nn.init.constant_(self.gate_conv.bias, float(gate_bias_init))

        # Default freeze proj-side
        self.freeze_proj()
        self._proj_unfrozen = False
        self.edge_text_adapter = None

    def enable_edge_enhance(self, dropout: float = 0.0):
        if self.edge_text_adapter is None:
            self.edge_text_adapter = EdgeTextResidualAdapter(self.d_latent, dropout=dropout)

    def edge_parameters(self):
        if self.edge_text_adapter is None:
            return []
        return list(self.edge_text_adapter.parameters())

    # ---- freeze / unfreeze proj group ----
    def freeze_proj(self):
        for p in self.visual_proj.parameters(): p.requires_grad_(False)
        self.T_anchor.requires_grad_(False)
        self.cls_bias.requires_grad_(False)
        self.log_tau.requires_grad_(False)
        if self.affinity_metric == 'hyperbolic':
            for p in self.v_pre_ln.parameters(): p.requires_grad_(False)
            self.rho.requires_grad_(False)
            self.log_tau_h.requires_grad_(False)
        if self.affinity_metric == 'hyperbolic_pathway' and self.hyp_pathway is not None:
            self.hyp_pathway.freeze_all()
        if self.affinity_metric == 'cma' and self.cma_block is not None:
            self.cma_block.freeze_all()
        for p in self.pooling.parameters(): p.requires_grad_(False)
        self._proj_unfrozen = False

    def unfreeze_proj(self):
        for p in self.visual_proj.parameters(): p.requires_grad_(True)
        self.T_anchor.requires_grad_(True)
        self.cls_bias.requires_grad_(True)
        self.log_tau.requires_grad_(True)
        if self.affinity_metric == 'hyperbolic':
            for p in self.v_pre_ln.parameters(): p.requires_grad_(True)
            self.rho.requires_grad_(True)
            self.log_tau_h.requires_grad_(True)
        if self.affinity_metric == 'hyperbolic_pathway' and self.hyp_pathway is not None:
            self.hyp_pathway.unfreeze_all()
        if self.affinity_metric == 'cma' and self.cma_block is not None:
            self.cma_block.unfreeze_all()
        if isinstance(self.pooling.theta, nn.Parameter):
            self.pooling.theta.requires_grad_(True)
        self._proj_unfrozen = True

    def proj_parameters(self):
        params = list(self.visual_proj.parameters()) + \
                 [self.T_anchor, self.cls_bias, self.log_tau] + \
                 list(self.pooling.parameters())
        if self.affinity_metric == 'hyperbolic':
            params += list(self.v_pre_ln.parameters()) + [self.rho, self.log_tau_h]
        if self.affinity_metric == 'hyperbolic_pathway' and self.hyp_pathway is not None:
            # All pathway parameters need an optimizer entry once unfrozen.
            params += list(self.hyp_pathway.parameters())
        if self.affinity_metric == 'cma' and self.cma_block is not None:
            params += list(self.cma_block.parameters())
        return params

    # ── hyperbolic helpers ─────────────────────────────────────────────
    def get_curvature(self) -> torch.Tensor:
        return F.softplus(self.rho) + self.hyperbolic_eps

    def get_tau_h(self) -> torch.Tensor:
        return F.softplus(self.log_tau_h) + self.hyperbolic_eps

    # ---- main forward: receives patches, returns prior + gate + cls ----
    def forward(self, patches: torch.Tensor, target_h: int, target_w: int,
                patch_hw=None, edge_prior: torch.Tensor = None):
        """
        patches:  [B, N, 768]   pre-computed last-layer DINOv2 patch tokens
                                   (from DPT main backbone, before any dropout)
        target_h, target_w: spatial size of the DPT logits we will fuse with.

        Returns dict:
            prior        [B, n_orig, target_h, target_w]
            gate         [B, 1,      target_h, target_w]   sigmoid output in [0,1]
            cls_logits   [B, C_new]                          image-level pooled
            non_bg_mask  [1, n_orig, 1, 1]                   1 for non-bg, 0 for bg
        """
        B, N, _ = patches.shape
        if patch_hw is None:
            side = int(round(N ** 0.5))
            patch_h, patch_w = side, side
        else:
            patch_h, patch_w = [int(v) for v in patch_hw]
        if patch_h * patch_w != N:
            raise RuntimeError(
                f"AffinitySideBranch patch grid mismatch: patch_h*patch_w={patch_h}*{patch_w} "
                f"but got N={N} patch tokens."
            )
        edge_tokens = None
        if edge_prior is not None:
            edge_tokens = edge_to_tokens(edge_prior, patch_h, patch_w)
        # IMPORTANT: read params via .clone() so multiple forward() calls in the same
        # training step (img_x + img_u_s1 + img_u_s2) don't share storage with the
        # original parameter tensors. This avoids version-counter conflicts when
        # the autograd engine saves the param for backward in several places.
        cls_bias = self.cls_bias.clone()                                # [C_new]

        if self.affinity_metric == 'cma':
            # Cross-Modal Adapter pathway (PR 2025):
            # adapter.forward_visual → Z_v;  adapter.forward_text → T (recomputed);
            # cosine attention → text aggregation → up_proj → LN → ReZero residual.
            out = self.cma_block(patches)
            p_lat = out['Z_v']
            if self.edge_text_adapter is not None and edge_tokens is not None:
                p_lat = self.edge_text_adapter(p_lat, edge_tokens)
                T = out['T']
                tau = out['tau'].clamp_min(1e-6)
                Zn = F.normalize(p_lat, dim=-1)
                Tn = F.normalize(T, dim=-1)
                sim = Zn @ Tn.t()
                attn = F.softmax(sim / tau, dim=-1)
                A = sim / tau + self.cma_block.cls_bias.view(1, 1, -1)
                text_per_patch = attn @ T
                eu_fused = self.cma_block.up_proj(text_per_patch)
                eu_fused = self.cma_block.fuse_ln(eu_fused)
                eu_fused = self.cma_block.fuse_drop(eu_fused)
                self.last_x_aware = patches + self.cma_block.rezero_gamma * eu_fused
            else:
                A = out['A']
                self.last_x_aware = out['X_aware']
        elif self.affinity_metric == 'hyperbolic_pathway':
            # Full hyperbolic feature pathway: returns Euclidean X_aware,
            # affinity A from fused latents, and tangent-space Z_v (reused
            # for the gate to avoid a second visual_proj forward — that
            # double-fire would make DDP raise "marked ready twice").
            out = self.hyp_pathway(patches)
            A      = out['A']                                            # [B, N, C']
            p_lat  = out['Z_v']                                           # [B, N, d_h]
            self.last_x_aware = out['X_aware']
        elif self.affinity_metric == 'hyperbolic':
            z_v  = self.v_pre_ln(self.visual_proj(patches))
            c    = self.get_curvature()
            tau  = self.get_tau_h()
            p_h  = expmap0(z_v, c=c, eps=self.hyperbolic_eps)
            t_h  = self.T_anchor.clone()
            A    = hyperbolic_affinity(p_h, t_h, tau_h=tau, class_bias=None,
                                        c=c, eps=self.hyperbolic_eps,
                                        distance_scale=self.hyperbolic_distance_scale)
            p_lat = z_v
            self.last_x_aware = None
        else:
            p_lat = F.normalize(self.visual_proj(patches), dim=-1)      # [B, N, d_lat]
            if self.edge_text_adapter is not None and edge_tokens is not None:
                p_lat = F.normalize(self.edge_text_adapter(p_lat, edge_tokens), dim=-1)
            T     = F.normalize(self.T_anchor.clone(), dim=-1)          # [C_new, d_lat]
            tau   = self.log_tau.clone().exp().clamp(1e-3, 1.0)
            A     = (p_lat @ T.T) / tau                                  # [B, N, C_new]

        pooled = self.pooling(A, class_bias=cls_bias, return_mask=True)
        cls_logits = pooled['image_logits']                             # [B, C_new]

        # Dense map (logits, not softmax): + bias for consistent units
        dense = A.permute(0, 2, 1).reshape(B, -1, patch_h, patch_w)     # [B, C_new, H_p, W_p]
        dense = dense + cls_bias.view(1, -1, 1, 1)
        dense_up = F.interpolate(dense, size=(target_h, target_w),
                                  mode='bilinear', align_corners=False)

        # Scatter into original DPT class index space
        idx = self.idx_orig_to_new.clamp(min=0)                         # bg→0 placeholder
        gathered = dense_up.index_select(dim=1, index=idx)              # [B, n_orig, H, W]
        prior_full = gathered * self.non_bg_mask                        # bg channels zeroed

        # Gate from same projected latent (reshaped to spatial)
        p_lat_2d = p_lat.permute(0, 2, 1).reshape(B, -1, patch_h, patch_w)  # [B, d_lat, H_p, W_p]
        gate_logits = self.gate_conv(p_lat_2d)                          # [B, 1, H_p, W_p]
        gate_up = F.interpolate(gate_logits, size=(target_h, target_w),
                                 mode='bilinear', align_corners=False)
        gate = torch.sigmoid(gate_up)

        return {
            'prior': prior_full,
            'gate':  gate,
            'cls_logits': cls_logits,
            'selection_mask': pooled['selection_mask'],
            'selected_count': pooled['selected_count'],
            'threshold': pooled['threshold'],
            'non_bg_mask': self.non_bg_mask,
        }

    def fuse(self, dpt_logits: torch.Tensor, side_out: dict) -> torch.Tensor:
        """Per-pixel gated fusion. Bg channels untouched (mask zeros their gate)."""
        eff_gate = side_out['gate'] * side_out['non_bg_mask']
        return (1.0 - eff_gate) * dpt_logits + eff_gate * side_out['prior']

    # ---- checkpoint helpers ----
    def state_dict_for_save(self):
        sd = {
            'visual_proj': self.visual_proj.state_dict(),
            'T_anchor':    self.T_anchor.detach().cpu(),
            'log_tau':     self.log_tau.detach().cpu(),
            'cls_bias':    self.cls_bias.detach().cpu(),
            'pooling_mode': self.pooling.mode,
            'topk':         self.pooling.topk,
            'threshold':    self.pooling.theta.detach().cpu(),
            'threshold_init': self.pooling.threshold_init,
            'threshold_gamma': self.pooling.gamma,
            'threshold_learnable': self.pooling.threshold_learnable,
            'min_selected': self.pooling.min_selected,
            'hard_threshold_eval': self.pooling.use_hard_threshold_eval,
            'gate_conv':   self.gate_conv.state_dict(),
            'proj_unfrozen': bool(self._proj_unfrozen),
            'idx_orig_to_new': self.idx_orig_to_new.detach().cpu(),
            'affinity_metric': self.affinity_metric,
            'hyperbolic_eps':            self.hyperbolic_eps,
            'hyperbolic_distance_scale': self.hyperbolic_distance_scale,
            'hyperbolic_dim':            self.d_hyperbolic,
        }
        if self.affinity_metric == 'hyperbolic':
            sd['hyperbolic_rho']       = self.rho.detach().cpu()
            sd['hyperbolic_log_tau_h'] = self.log_tau_h.detach().cpu()
            sd['hyperbolic_v_pre_ln']  = self.v_pre_ln.state_dict()
        if self.affinity_metric == 'hyperbolic_pathway' and self.hyp_pathway is not None:
            sd['hyp_pathway_state'] = self.hyp_pathway.state_dict_for_save()
        if self.affinity_metric == 'cma' and self.cma_block is not None:
            sd['cma_block_state'] = self.cma_block.state_dict_for_save()
        if self.edge_text_adapter is not None:
            sd['edge_text_adapter'] = self.edge_text_adapter.state_dict()
        return sd

    def load_state_dict_from_save(self, sd):
        self.visual_proj.load_state_dict(sd['visual_proj'])
        self.T_anchor.data.copy_(sd['T_anchor'])
        self.log_tau.data.copy_(sd['log_tau'])
        self.cls_bias.data.copy_(sd['cls_bias'])
        if 'threshold' in sd and sd['threshold'] is not None:
            self.pooling.theta.data.copy_(sd['threshold'])
        self.gate_conv.load_state_dict(sd['gate_conv'])
        if self.affinity_metric == 'hyperbolic':
            if sd.get('hyperbolic_v_pre_ln') is not None:
                self.v_pre_ln.load_state_dict(sd['hyperbolic_v_pre_ln'])
            if sd.get('hyperbolic_rho') is not None:
                self.rho.data.copy_(sd['hyperbolic_rho'])
            if sd.get('hyperbolic_log_tau_h') is not None:
                self.log_tau_h.data.copy_(sd['hyperbolic_log_tau_h'])
        if (self.affinity_metric == 'hyperbolic_pathway'
                and self.hyp_pathway is not None
                and sd.get('hyp_pathway_state') is not None):
            self.hyp_pathway.load_state_dict_from_save(sd['hyp_pathway_state'])
        if (self.affinity_metric == 'cma'
                and self.cma_block is not None
                and sd.get('cma_block_state') is not None):
            self.cma_block.load_state_dict_from_save(sd['cma_block_state'])
        if self.edge_text_adapter is not None and sd.get('edge_text_adapter') is not None:
            self.edge_text_adapter.load_state_dict(sd['edge_text_adapter'])
        if sd.get('proj_unfrozen', False):
            self.unfreeze_proj()
        else:
            self.freeze_proj()

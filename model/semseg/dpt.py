import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.backbone.dinov2 import DINOv2
from model.visual_adapter import DinoDPTAdapter
from model.visual_adapter_moe import DinoDPTAdapterMoE
from model.util.blocks import FeatureFusionBlock, _make_scratch


def _make_fusion_block(features, use_bn, size=None):
    return FeatureFusionBlock(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
        size=size,
    )


class DPTHead(nn.Module):
    def __init__(
        self, 
        nclass,
        in_channels, 
        features=256, 
        use_bn=False, 
        out_channels=[256, 512, 1024, 1024],
    ):
        super(DPTHead, self).__init__()
        
        self.projects = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channel,
                kernel_size=1,
                stride=1,
                padding=0,
            ) for out_channel in out_channels
        ])
        
        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(
                in_channels=out_channels[0],
                out_channels=out_channels[0],
                kernel_size=4,
                stride=4,
                padding=0),
            nn.ConvTranspose2d(
                in_channels=out_channels[1],
                out_channels=out_channels[1],
                kernel_size=2,
                stride=2,
                padding=0),
            nn.Identity(),
            nn.Conv2d(
                in_channels=out_channels[3],
                out_channels=out_channels[3],
                kernel_size=3,
                stride=2,
                padding=1)
        ])
        
        self.scratch = _make_scratch(
            out_channels,
            features,
            groups=1,
            expand=False,
        )
        
        self.scratch.stem_transpose = None
        
        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)

        self.scratch.output_conv = nn.Sequential(
            nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(features, nclass, kernel_size=1, stride=1, padding=0)
        )
    
    def forward(self, out_features, patch_h, patch_w):
        out = []
        for i, x in enumerate(out_features):
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))
            
            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            
            out.append(x)
        
        layer_1, layer_2, layer_3, layer_4 = out
        
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)
        
        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])        
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)
        
        out = self.scratch.output_conv(path_1)
        
        return out


class DPT(nn.Module):
    def __init__(
        self, 
        encoder_size='base', 
        nclass=21,
        features=128, 
        out_channels=[96, 192, 384, 768], 
        use_bn=False,
    ):
        super(DPT, self).__init__()
        
        self.intermediate_layer_idx = {
            'small': [2, 5, 8, 11],
            'base': [2, 5, 8, 11], 
            'large': [4, 11, 17, 23], 
            'giant': [9, 19, 29, 39]
        }
        
        self.encoder_size = encoder_size
        self.backbone = DINOv2(model_name=encoder_size)
        
        self.head = DPTHead(nclass, self.backbone.embed_dim, features, use_bn, out_channels=out_channels)
        self.visual_adapter = None
        self.rein = None
        
        self.binomial = torch.distributions.binomial.Binomial(probs=0.5)
        
    def lock_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def enable_visual_adapter(self, reduction=8, dropout=0.0, moe=False,
                              moe_edge_cond=False, moe_text_cond_dim=0):
        n_feat = len(self.intermediate_layer_idx[self.encoder_size])
        if moe:
            self._va_is_moe = True
            self._moe_edge_cond_dim = 1 if moe_edge_cond else 0
            self._moe_text_cond_dim = int(moe_text_cond_dim)
            self._text_cond = None
            self.visual_adapter = DinoDPTAdapterMoE(
                self.backbone.embed_dim, num_layers=n_feat,
                reduction=reduction, dropout=dropout,
                text_cond_dim=self._moe_text_cond_dim,
                edge_cond_dim=self._moe_edge_cond_dim,
            )
        else:
            self._va_is_moe = False
            self.visual_adapter = DinoDPTAdapter(
                self.backbone.embed_dim, num_layers=n_feat,
                reduction=reduction, dropout=dropout,
            )

    def set_text_cond(self, text_cond):
        """Legacy external setter for MoE text conditioning.

        The training/eval path now prefers ``forward(..., moe_text_cond_fn=...)``
        so the condition is generated from the current raw patch tokens inside
        the same forward pass.
        """
        self._text_cond = text_cond

    def enable_rein(self, num_tokens=100, token_rank=16, token_dim=64,
                    dropout=0.0, adapt_layers=None):
        """Rein-style in-backbone PEFT (see model/rein_adapter.py).

        The ReinAdapter is registered on the DPT (name 'rein.*') so it lands in
        the lr*lr_multi param group and is NOT frozen by lock_backbone(). The
        frozen backbone gets a NON-registered reference (`object.__setattr__`)
        so the block loop can call it without adding it to backbone.parameters()
        and so a single deepcopy(model) (the EMA teacher) shares one copy.
        """
        from model.rein_adapter import ReinAdapter
        self.rein = ReinAdapter(
            self.backbone.embed_dim, self.backbone.n_blocks,
            num_tokens=num_tokens, token_rank=token_rank, token_dim=token_dim,
            dropout=dropout, adapt_layers=adapt_layers,
        )
        object.__setattr__(self.backbone, '_rein', self.rein)
        return self.rein

    def adapt_features(self, features, patch_h, patch_w, text_cond=None, edge_cond=None):
        if self.visual_adapter is None:
            return features
        if getattr(self, '_va_is_moe', False):
            return self.visual_adapter(features, patch_h, patch_w,
                                       text_cond=text_cond, edge_cond=edge_cond)
        return self.visual_adapter(features, patch_h, patch_w)
    
    # ── New: split into encode (backbone) + decode (DPT head) ──
    def encode(self, x, return_cls=False):
        """Run DINOv2 backbone only.

        Returns:
            features:   list of intermediate-layer patch token tensors [B, N, embed_dim]
            cls_tok:    last-layer CLS token  [B, embed_dim]   (None if return_cls=False)
            patch_h:    H // 14
            patch_w:    W // 14
        """
        patch_h, patch_w = x.shape[-2] // 14, x.shape[-1] // 14
        if return_cls:
            out_pairs = self.backbone.get_intermediate_layers(
                x, self.intermediate_layer_idx[self.encoder_size],
                return_class_token=True,
            )
            features = [p for p, _ in out_pairs]
            cls_tok = out_pairs[-1][1]
        else:
            features = self.backbone.get_intermediate_layers(
                x, self.intermediate_layer_idx[self.encoder_size]
            )
            cls_tok = None
        return features, cls_tok, patch_h, patch_w

    def decode(self, features, patch_h, patch_w, comp_drop=False):
        """Run DPT head + bilinear up to native crop size.  features is a list
        of intermediate patch-token tensors [B, N, embed_dim]."""
        if comp_drop:
            features = list(features)
            bs, dim = features[0].shape[0], features[0].shape[-1]
            device = features[0].device
            dropout_mask1 = self.binomial.sample((bs // 2, dim)).to(device) * 2.0
            dropout_mask2 = 2.0 - dropout_mask1
            dropout_prob = 0.5
            num_kept = int(bs // 2 * (1 - dropout_prob))
            kept_indexes = torch.randperm(bs // 2)[:num_kept]
            dropout_mask1[kept_indexes, :] = 1.0
            dropout_mask2[kept_indexes, :] = 1.0
            dropout_mask = torch.cat((dropout_mask1, dropout_mask2))
            features = [feature * dropout_mask.unsqueeze(1) for feature in features]
        out = self.head(features, patch_h, patch_w)
        out = F.interpolate(out, (patch_h * 14, patch_w * 14),
                             mode='bilinear', align_corners=True)
        return out

    def forward(self, x, comp_drop=False, return_cls=False, patches_override=None,
                edge_prior=None, moe_text_cond_fn=None):
        """Forward DPT.

        If ``patches_override`` is provided (shape [B, N, embed_dim] matching the
        last intermediate feature), it REPLACES features[-1] before the head.
        This is the integration point for the hyperbolic feature pathway:

            features, cls, ph, pw = self.encode(x, return_cls=True)
            X_aware = hyp_pathway(features[-1])['X_aware']
            features[-1] = X_aware              # ← what patches_override does
            out = self.decode(features, ph, pw)

        Existing callers that don't pass `patches_override` get IDENTICAL
        behavior to the original forward (backward-compatible).
        """
        features, cls_tok, patch_h, patch_w = self.encode(x, return_cls=return_cls)
        _moe_text_cond = None
        if (getattr(self, '_va_is_moe', False)
                and getattr(self, '_moe_text_cond_dim', 0) > 0
                and moe_text_cond_fn is not None):
            _moe_text_cond = moe_text_cond_fn(features[-1], patch_h, patch_w, edge_prior)
            if _moe_text_cond is not None:
                expected = int(getattr(self, '_moe_text_cond_dim', 0))
                if (_moe_text_cond.dim() != 3
                        or _moe_text_cond.shape[:2] != features[-1].shape[:2]
                        or _moe_text_cond.shape[-1] != expected):
                    raise RuntimeError(
                        "HPTA-MoE text condition must have shape "
                        f"[B, N, {expected}], got {tuple(_moe_text_cond.shape)}"
                    )
        # MoE visual adapter: build the per-token edge signal (self-contained from
        # the RGB edge prior) so the router coordinates with the edge module; the
        # text signal is supplied by the LC-PAM branch via moe_text_cond_fn.
        _moe_edge_cond = None
        if getattr(self, '_va_is_moe', False) and getattr(self, '_moe_edge_cond_dim', 0) > 0:
            from util.edge_enhance import edge_to_tokens, rgb_edge_prior
            _ep = edge_prior.detach() if edge_prior is not None else rgb_edge_prior(x)
            _et = edge_to_tokens(_ep, patch_h, patch_w)
            if _et.dim() == 3 and _et.shape[-1] != 1:
                _et = _et.mean(dim=-1, keepdim=True)
            _moe_edge_cond = _et
        features = self.adapt_features(
            features, patch_h, patch_w,
            text_cond=(_moe_text_cond if _moe_text_cond is not None
                       else getattr(self, '_text_cond', None)),
            edge_cond=_moe_edge_cond)
        # patches BEFORE comp_drop / override, returned to caller for side branch
        clean_patches = features[-1] if return_cls else None

        edge_boundary_logits = None
        if patches_override is not None:
            assert patches_override.shape == features[-1].shape, (
                f'patches_override shape {tuple(patches_override.shape)} != '
                f'features[-1] shape {tuple(features[-1].shape)}'
            )
            features = list(features)
            features[-1] = patches_override
        elif hasattr(self, 'edge_seg_adapter'):
            from util.edge_enhance import edge_to_tokens, rgb_edge_prior
            if edge_prior is None:
                edge_prior = rgb_edge_prior(x)
            edge_tokens = edge_to_tokens(edge_prior, patch_h, patch_w)
            features = list(features)
            features[-1], edge_boundary_logits = self.edge_seg_adapter(
                features[-1], edge_tokens, patch_h, patch_w)

        out = self.decode(features, patch_h, patch_w, comp_drop=comp_drop)
        self.last_edge_boundary_logits = edge_boundary_logits
        if return_cls:
            return out, cls_tok, clean_patches
        return out

"""Class-prior debiasing for SSL pseudo-label generation.

Inspired by DiffMatch (ICLR 2025) and logit adjustment (Menon et al. ICLR 2021).

Core observation: when generating pseudo-labels from an EMA teacher via
``argmax(softmax(logits)) > conf_thresh``, head classes dominate because:
  1) the teacher is more confident on majority classes (it sees more of them),
  2) the confidence threshold removes most tail-class predictions,
  3) the resulting pseudo-labeled distribution feeds back into training, so
     the model becomes EVEN more biased toward heads next epoch.

Fix: before applying argmax / threshold, subtract ``tau * log pi_c`` from
each class logit, where ``pi_c`` is the per-class prior. Equivalently, divide
the softmax by ``pi_c ** tau``. This pulls rare-class logits up and
common-class logits down, so the resulting pseudo-labels are class-prior-debiased.

Two ways to estimate ``pi_c``:
  - **fixed**: compute once from the labeled training set and freeze it
    (cheap, often good enough).
  - **running**: an EMA of class-frequency-from-pseudo-labels updated every
    iteration (DebiasMatch / FreeMatch style, adapts as training proceeds).

We provide ``PriorEstimator`` to keep a running prior and a stateless
``adjust_logits`` for inference.

Usage in an SSL trainer:

    # 1) construct (once, after dataset is known)
    prior = PriorEstimator(num_classes=C, decay=0.99,
                            init_from_labeled='/path/to/labeled.txt',
                            num_classes_check=C, ignore_index=255)
    prior.to(device)

    # 2) in training loop, when generating pseudo from teacher:
    with torch.no_grad():
        logits_t = teacher(weak_u)['semantic_logits']
        adj = adjust_logits(logits_t, prior.get_log_prior(), tau=cfg.debias.tau)
        prob = F.softmax(adj, dim=1)
        conf, pseudo = prob.max(dim=1)
        valid = conf >= conf_thr

    # 3) update prior with the *raw* (un-debiased) pseudo-counts so it stays
    #    representative of the data distribution, not the debiased decisions:
    prior.update_from_pseudo(pseudo_raw=raw_argmax, valid=valid)
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional

import torch


def adjust_logits(logits: torch.Tensor, log_prior: torch.Tensor,
                  tau: float = 1.0) -> torch.Tensor:
    """Subtract ``tau * log_prior`` from each class logit.

    Args:
        logits:    [B, C, *spatial] raw logits.
        log_prior: [C] log of class prior pi_c (clamped against zero).
        tau:       strength scalar in [0, ~2]. tau=1.0 = standard Menon adjustment.

    Returns:
        adjusted logits, same shape as ``logits``.
    """
    if tau == 0.0:
        return logits
    # broadcast [C] -> [1, C, 1, 1] (or however many spatial dims).
    shape = [1, -1] + [1] * (logits.ndim - 2)
    return logits - tau * log_prior.to(logits.dtype).view(*shape)


class PriorEstimator(torch.nn.Module):
    """Tracks per-class frequency pi_c as a running EMA over pseudo-label counts.

    Stored as a non-trainable buffer so DDP/checkpointing work out of the box.

    Args:
        num_classes: C.
        decay:       EMA momentum for the running prior. 0.99 ≈ 100-iter window.
                     0.0 disables updates (frozen prior).
        eps:         minimum probability per class (prevents -inf in log).
        init_from_labeled: optional path to labeled.txt (lines: ``img\\tmask``);
                     if given, the prior is initialized by counting class
                     pixels in that split. Otherwise initialized uniform.
        ignore_index: int label value to skip during init counting.
    """
    def __init__(self, num_classes: int, decay: float = 0.99, eps: float = 1e-4,
                 init_from_labeled: Optional[str] = None,
                 data_root: Optional[str] = None,
                 ignore_index: int = 255):
        super().__init__()
        self.C = int(num_classes)
        self.decay = float(decay)
        self.eps = float(eps)
        self.register_buffer('prior', torch.full((self.C,),
                                                  1.0 / self.C))
        self.register_buffer('log_prior', torch.log(self.prior.clamp(min=self.eps)))

        if init_from_labeled and Path(init_from_labeled).exists():
            self._init_from_split(init_from_labeled, ignore_index, data_root)

    def _init_from_split(self, split_path: str, ignore_index: int,
                          data_root: Optional[str]) -> None:
        from PIL import Image
        import numpy as np
        counts = torch.zeros(self.C, dtype=torch.float64)
        n_files = 0
        n_skipped = 0
        root = Path(data_root) if data_root else None
        for line in Path(split_path).read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t') if '\t' in line else line.split()
            if len(parts) < 2:
                continue
            mask_rel = parts[1]
            mp = Path(mask_rel)
            if not mp.is_absolute():
                if root is not None:
                    mp = root / mp
                else:
                    n_skipped += 1
                    continue
            if not mp.exists():
                n_skipped += 1
                continue
            try:
                arr = np.array(Image.open(mp))
            except Exception:
                n_skipped += 1
                continue
            arr = arr.reshape(-1)
            arr = arr[arr != ignore_index]
            for c in range(self.C):
                counts[c] += int((arr == c).sum())
            n_files += 1
            if n_files >= 500:
                break
        if counts.sum() > 0:
            prior = (counts / counts.sum()).float()
            self.prior.copy_(prior.clamp(min=self.eps))
            self.log_prior.copy_(torch.log(self.prior))
            print(f'[PriorEstimator] init from {n_files} masks '
                  f'(skipped {n_skipped}): '
                  f'{[round(float(p), 4) for p in self.prior.tolist()]}')
        else:
            print(f'[PriorEstimator] no usable masks in {split_path} '
                  f'(data_root={data_root}); using uniform prior.')

    @torch.no_grad()
    def update_from_pseudo(self, pseudo_raw: torch.Tensor,
                           valid: Optional[torch.Tensor] = None,
                           sync_ddp: bool = False) -> None:
        """Update running prior with class counts from one batch's pseudo-labels.

        Args:
            pseudo_raw: [B, *] long tensor of class indices.
            valid:      [B, *] bool tensor; ignored where False.
            sync_ddp:   if True and torch.distributed is initialised, sum class
                         counts across all ranks before computing the batch prior
                         (otherwise each rank drifts independently).
        """
        if self.decay <= 0.0:
            return
        if valid is not None:
            pseudo_raw = pseudo_raw[valid]
        counts = torch.bincount(pseudo_raw.view(-1).long(),
                                 minlength=self.C).float()
        if sync_ddp:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(counts, op=dist.ReduceOp.SUM)
        total = counts.sum()
        if total <= 0:
            return
        batch_prior = counts / total.clamp(min=1.0)
        self.prior.mul_(self.decay).add_(batch_prior.to(self.prior),
                                           alpha=1.0 - self.decay)
        self.prior.clamp_(min=self.eps)
        self.prior.div_(self.prior.sum())
        self.log_prior.copy_(torch.log(self.prior))

    def get_log_prior(self) -> torch.Tensor:
        return self.log_prior

    def get_prior(self) -> torch.Tensor:
        return self.prior

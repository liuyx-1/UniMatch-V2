# Three-Module Ablation Runbook (HPTA + LC-PAM + MGER-guided routing + TCR)

Corresponds to `Table abl` in `paper/JBHI_LaTex_Template/method_experiments.tex`:
base (UniMatch-V2) / +text (HPTA+LC-PAM) / +edge (+MGER-guided routing) / +full (+TCR).

TS-MDR and the instance-aware loss are explicitly **disabled** in every row
here (`--no-tsmdr --no-instance-loss --no-instance-loss-unlabeled`) so this
sweep isolates exactly the four classic modules — they get their own
ablation axis separately (see `method_experiments.tex` §Ablation Study).

## Batch size: empirically confirmed, not guessed

This server had never run our own `unimatch_v2.py` trainer before (only
AllSpark/BCP competitor code), so there was no precedent batch size for this
box. I ran a live probe of the heaviest row (`full`: HPTA+LC-PAM+MGER-routing+TCR,
Endoscapes-Seg50, crop 490, r=0.10) at **BS=12** on the freed 24GB RTX 3090:

- Peak memory: **~22.4–22.7 GB / 24.5 GB** (training steps and, critically,
  full-resolution validation eval — the more memory-hungry of the two) held
  stable through epoch 10+ with no OOM.
- Only ~2GB headroom — tight but proven stable across both train and eval.
  **BS=12 is used for all four rows** for a fair comparison; the lighter
  rows (base/+text, no edge/temporal forward passes) will have *more*
  headroom than what was actually observed, so they're safe by construction.
- `scripts/train.sh` has a built-in `AUTO_BS=1` safety net (halves batch
  size automatically on detected CUDA OOM, default enabled) — a genuine OOM
  in a lighter row would self-correct rather than crash the whole queue.

If you want to push batch size higher on a future run: try BS=16 with
`AUTO_BS=1` left on (default) so it safely falls back if it doesn't fit —
do **not** disable the OOM auto-retry when experimenting upward.

## Anchor datasets & environment

Per the paper's ablation protocol: **Endoscapes-Seg50** and **EndoVis2018**,
both at **r=0.10** (labeled ratio), 80 epochs (matches each dataset's config
default).

Required env setup (SigLIP-2 / DINOv2 weights are cached locally, but
`transformers` still does a HEAD-request version check unless forced
offline — this caused an OOM-unrelated hang on first attempt):
```bash
export PATH=/root/autodl-tmp/envs/unimatchv2/bin:$PATH
source /etc/network_turbo 2>/dev/null || true
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
```

## Status as of 2026-07-02 ~01:19 CST

| Dataset | Row | Status |
|---|---|---|
| Endoscapes-Seg50 | base | queued (`module_ablation_queue` screen) |
| Endoscapes-Seg50 | +text | queued |
| Endoscapes-Seg50 | +edge | queued |
| Endoscapes-Seg50 | **+full (=Ours)** | **already running** — `probe_full_bs12`, port 29601, started 01:07, epoch 10/80 as of this note, ETA ~53 min |
| EndoVis2018 | base | queued |
| EndoVis2018 | +text | queued |
| EndoVis2018 | +edge | queued |
| EndoVis2018 | +full (=Ours) | queued |

All 7 remaining rows are running sequentially (single GPU) in the
`module_ablation_queue` screen session, launched via
`scripts/run_module_ablation.sh` (waits for free GPU before each row, same
pattern as the BCP queue script). Safe to leave unattended.

```bash
screen -r module_ablation_queue      # attach to watch live
# Ctrl-A D to detach without killing it
tail -f /root/autodl-tmp/code/UniMatch-V2_local/logs/module_ablation_queue.log
```

## Manual commands (if you want to (re)run a single row by hand)

```bash
cd /root/autodl-tmp/code/UniMatch-V2_local
export PATH=/root/autodl-tmp/envs/unimatchv2/bin:$PATH
source /etc/network_turbo 2>/dev/null || true
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false

# base (no modules)
RATE=0.10 BS=12 EPOCHS=80 \
  EXTRA='--no-tsmdr --no-instance-loss --no-instance-loss-unlabeled' \
  TAG=ablation_base_r0.10 \
  bash scripts/train.sh 1 29610 base endoscapes_seg50

# +text (HPTA-MoE + LC-PAM text-conditioned routing)
RATE=0.10 BS=12 EPOCHS=80 VISUAL_ADAPTER=1 VA_MOE=1 JOINT_TEXT_STAGE=1 MOE_TEXT_COND_DIM=128 \
  EXTRA='--no-tsmdr --no-instance-loss --no-instance-loss-unlabeled' \
  TAG=ablation_text_r0.10 \
  bash scripts/train.sh 1 29611 affinity_min endoscapes_seg50

# +edge (HPTA-MoE + LC-PAM + MGER-guided edge routing)
RATE=0.10 BS=12 EPOCHS=80 VISUAL_ADAPTER=1 VA_MOE=1 JOINT_TEXT_STAGE=1 MOE_TEXT_COND_DIM=128 \
  EDGE_ENHANCE=1 MOE_EDGE_COND=1 EDGE_REFINER=1 \
  EXTRA='--no-tsmdr --no-instance-loss --no-instance-loss-unlabeled' \
  TAG=ablation_edge_r0.10 \
  bash scripts/train.sh 1 29612 base endoscapes_seg50

# +full = Ours (HPTA-MoE + LC-PAM + MGER-guided edge routing + TCR)
RATE=0.10 BS=12 EPOCHS=80 VISUAL_ADAPTER=1 VA_MOE=1 JOINT_TEXT_STAGE=1 MOE_TEXT_COND_DIM=128 \
  EDGE_ENHANCE=1 MOE_EDGE_COND=1 EDGE_REFINER=1 \
  EXTRA='--no-tsmdr --no-instance-loss --no-instance-loss-unlabeled' \
  TAG=ablation_full_r0.10 \
  bash scripts/train.sh 1 29613 full endoscapes_seg50
```
Swap `endoscapes_seg50` → `endovis2018` (config auto-switches crop/classes)
for the second anchor dataset. Each `save_path` is
`/root/autodl-tmp/exp/<dataset>/unimatch_v2_<TAG>/`, containing `best.pth`,
`latest.pth`, and a timestamped training log with per-epoch mIoU.

## Extending to TS-MDR / instance-loss ablation axes (separate sweep)

Once this 4-row sweep is done, the tex's Table abl also wants two more rows
(`+TS-MDR`, `+instance-loss L`, `+instance-loss L+U` — see the 7-row table
in `method_experiments.tex`). Reuse the same BS=12 and just drop the
`--no-tsmdr` / `--no-instance-loss*` flags progressively instead of the
`full` row's `EXTRA`:
```bash
# + TS-MDR on top of full
RATE=0.10 BS=12 EPOCHS=80 VISUAL_ADAPTER=1 VA_MOE=1 JOINT_TEXT_STAGE=1 MOE_TEXT_COND_DIM=128 \
  EDGE_ENHANCE=1 MOE_EDGE_COND=1 EDGE_REFINER=1 \
  TSMDR=1 \
  EXTRA='--no-instance-loss --no-instance-loss-unlabeled' \
  TAG=ablation_tsmdr_r0.10 \
  bash scripts/train.sh 1 29614 full endoscapes_seg50
```
(instance-loss is on by default now — no extra flag needed once you stop
passing `--no-instance-loss`.)

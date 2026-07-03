# SASR Ablation Commands

This file lists the 16 ablation runs for **Endoscapes-Seg50** and
**EndoVis2018** at label ratio `0.10`.

Edit the variables in the setup block before launching:

```bash
cd /root/autodl-tmp/code/UniMatch-V2_local
export PATH=/root/autodl-tmp/envs/unimatchv2/bin:$PATH
source /etc/network_turbo 2>/dev/null || true
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
RATE=0.10
BS=12
LR=5e-5
EPOCHS=80
CROP=490
NGPU=1
```

Rows:

| Row | Experiment |
|---|---|
| 1 | UniMatchV2 official |
| 2 | Base + HPTA |
| 3 | Base + HPTA-MoE |
| 4 | Base + HPTA-MoE + LC-PAM |
| 5 | Base + HPTA-MoE + MGER |
| 6 | Base + HPTA-MoE + SMCR/TMRC |
| 7 | Base + HPTA-MoE + Instance-aware loss |
| 8 | Full |

Note: current code still uses the flag name `ROUTED_CONSISTENCY` for the
consistency module. In the paper naming, this corresponds to **SMCR**.

## Endoscapes-Seg50

```bash
RATE=$RATE BS=$BS LR=$LR EPOCHS=$EPOCHS CROP=$CROP TAG=abl01_base_r${RATE} EXTRA='--no-tsmdr --no-instance-loss --no-instance-loss-unlabeled' bash scripts/train.sh $NGPU 29610 base endoscapes_seg50
```

```bash
RATE=$RATE BS=$BS LR=$LR EPOCHS=$EPOCHS CROP=$CROP VISUAL_ADAPTER=1 TAG=abl02_hpta_r${RATE} EXTRA='--no-tsmdr --no-instance-loss --no-instance-loss-unlabeled' bash scripts/train.sh $NGPU 29611 base endoscapes_seg50
```

```bash
RATE=$RATE BS=$BS LR=$LR EPOCHS=$EPOCHS CROP=$CROP VISUAL_ADAPTER=1 VA_MOE=1 TAG=abl03_hpta_moe_r${RATE} EXTRA='--no-tsmdr --no-instance-loss --no-instance-loss-unlabeled' bash scripts/train.sh $NGPU 29612 base endoscapes_seg50
```

```bash
RATE=$RATE BS=$BS LR=$LR EPOCHS=$EPOCHS CROP=$CROP VISUAL_ADAPTER=1 VA_MOE=1 JOINT_TEXT_STAGE=1 MOE_TEXT_COND_DIM=128 TAG=abl04_hpta_moe_lcpam_r${RATE} EXTRA='--no-tsmdr --no-instance-loss --no-instance-loss-unlabeled' bash scripts/train.sh $NGPU 29613 affinity_min endoscapes_seg50
```

```bash
RATE=$RATE BS=$BS LR=$LR EPOCHS=$EPOCHS CROP=$CROP VISUAL_ADAPTER=1 VA_MOE=1 EDGE_ENHANCE=1 EDGE_REFINER=1 MOE_EDGE_COND=1 TAG=abl05_hpta_moe_mger_r${RATE} EXTRA='--no-tsmdr --no-instance-loss --no-instance-loss-unlabeled' bash scripts/train.sh $NGPU 29614 base endoscapes_seg50
```

```bash
RATE=$RATE BS=$BS LR=$LR EPOCHS=$EPOCHS CROP=$CROP VISUAL_ADAPTER=1 VA_MOE=1 ROUTED_CONSISTENCY=1 ROUTE=text CONSISTENCY_WEIGHT=0.1 CONSISTENCY_BETA=0.1 CONSISTENCY_WARMUP=10 TAG=abl06_hpta_moe_smcr_r${RATE} EXTRA='--no-tsmdr --no-instance-loss --no-instance-loss-unlabeled' bash scripts/train.sh $NGPU 29615 base endoscapes_seg50
```

```bash
RATE=$RATE BS=$BS LR=$LR EPOCHS=$EPOCHS CROP=$CROP VISUAL_ADAPTER=1 VA_MOE=1 TAG=abl07_hpta_moe_iasl_r${RATE} EXTRA='--no-tsmdr' bash scripts/train.sh $NGPU 29616 base endoscapes_seg50
```

```bash
RATE=$RATE BS=$BS LR=$LR EPOCHS=$EPOCHS CROP=$CROP VISUAL_ADAPTER=1 VA_MOE=1 JOINT_TEXT_STAGE=1 MOE_TEXT_COND_DIM=128 EDGE_ENHANCE=1 EDGE_REFINER=1 MOE_EDGE_COND=1 ROUTED_CONSISTENCY=1 ROUTE=text CONSISTENCY_WEIGHT=0.1 CONSISTENCY_BETA=0.1 CONSISTENCY_WARMUP=10 TAG=abl08_full_r${RATE} bash scripts/train.sh $NGPU 29617 affinity_min endoscapes_seg50
```

## EndoVis2018

```bash
RATE=$RATE BS=$BS LR=$LR EPOCHS=$EPOCHS CROP=$CROP TAG=abl01_base_r${RATE} EXTRA='--no-tsmdr --no-instance-loss --no-instance-loss-unlabeled' bash scripts/train.sh $NGPU 29620 base endovis2018
```

```bash
RATE=$RATE BS=$BS LR=$LR EPOCHS=$EPOCHS CROP=$CROP VISUAL_ADAPTER=1 TAG=abl02_hpta_r${RATE} EXTRA='--no-tsmdr --no-instance-loss --no-instance-loss-unlabeled' bash scripts/train.sh $NGPU 29621 base endovis2018
```

```bash
RATE=$RATE BS=$BS LR=$LR EPOCHS=$EPOCHS CROP=$CROP VISUAL_ADAPTER=1 VA_MOE=1 TAG=abl03_hpta_moe_r${RATE} EXTRA='--no-tsmdr --no-instance-loss --no-instance-loss-unlabeled' bash scripts/train.sh $NGPU 29622 base endovis2018
```

```bash
RATE=$RATE BS=$BS LR=$LR EPOCHS=$EPOCHS CROP=$CROP VISUAL_ADAPTER=1 VA_MOE=1 JOINT_TEXT_STAGE=1 MOE_TEXT_COND_DIM=128 TAG=abl04_hpta_moe_lcpam_r${RATE} EXTRA='--no-tsmdr --no-instance-loss --no-instance-loss-unlabeled' bash scripts/train.sh $NGPU 29623 affinity_min endovis2018
```

```bash
RATE=$RATE BS=$BS LR=$LR EPOCHS=$EPOCHS CROP=$CROP VISUAL_ADAPTER=1 VA_MOE=1 EDGE_ENHANCE=1 EDGE_REFINER=1 MOE_EDGE_COND=1 TAG=abl05_hpta_moe_mger_r${RATE} EXTRA='--no-tsmdr --no-instance-loss --no-instance-loss-unlabeled' bash scripts/train.sh $NGPU 29624 base endovis2018
```

```bash
RATE=$RATE BS=$BS LR=$LR EPOCHS=$EPOCHS CROP=$CROP VISUAL_ADAPTER=1 VA_MOE=1 ROUTED_CONSISTENCY=1 ROUTE=text CONSISTENCY_WEIGHT=0.1 CONSISTENCY_BETA=0.1 CONSISTENCY_WARMUP=10 TAG=abl06_hpta_moe_smcr_r${RATE} EXTRA='--no-tsmdr --no-instance-loss --no-instance-loss-unlabeled' bash scripts/train.sh $NGPU 29625 base endovis2018
```

```bash
RATE=$RATE BS=$BS LR=$LR EPOCHS=$EPOCHS CROP=$CROP VISUAL_ADAPTER=1 VA_MOE=1 TAG=abl07_hpta_moe_iasl_r${RATE} EXTRA='--no-tsmdr' bash scripts/train.sh $NGPU 29626 base endovis2018
```

```bash
RATE=$RATE BS=$BS LR=$LR EPOCHS=$EPOCHS CROP=$CROP VISUAL_ADAPTER=1 VA_MOE=1 JOINT_TEXT_STAGE=1 MOE_TEXT_COND_DIM=128 EDGE_ENHANCE=1 EDGE_REFINER=1 MOE_EDGE_COND=1 ROUTED_CONSISTENCY=1 ROUTE=text CONSISTENCY_WEIGHT=0.1 CONSISTENCY_BETA=0.1 CONSISTENCY_WARMUP=10 TAG=abl08_full_r${RATE} bash scripts/train.sh $NGPU 29627 affinity_min endovis2018
```

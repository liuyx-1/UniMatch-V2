# SASR

**Structure-Aware Semantic Routing for Semi-Supervised Surgical Segmentation**

This repository contains the cleaned implementation of SASR for
semi-supervised surgical segmentation. The codebase is built on UniMatch-V2 and
adds the current SASR components:

- **HPTA-MoE**: lightweight visual adaptation with mixture-of-experts routing.
- **LC-PAM**: language-patch affinity module for semantic guidance.
- **MGER**: morphology-guided edge refinement for structural guidance.
- **SMCR/TMRC**: semantic-morphology consistency regularization implemented in
  code as routed consistency.
- **IASL**: instance-aware structural loss.

## Main Entrypoints

- Training: `unimatch_v2.py`
- Evaluation: `test.py`
- Main train wrapper: `scripts/train.sh`
- Main test wrapper: `scripts/test.sh`
- Ablation commands: `docs/SASR_ABLATION_COMMANDS.md`

## Quick Setup

```bash
cd /root/autodl-tmp/code/SASR
export PATH=/root/autodl-tmp/envs/unimatchv2/bin:$PATH
source /etc/network_turbo 2>/dev/null || true
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
```

Install dependencies according to the local environment:

```bash
pip install -r requirements.txt
```

Backbone checkpoints, datasets, splits, and experiment outputs are not tracked
in this repository. Place them under the paths expected by `configs/*.yaml` and
the shell scripts, or override the paths with environment variables.

## Ablation Experiments

The current 16-command ablation plan covers:

- Endoscapes-Seg50 at `RATE=0.10`
- EndoVis2018 at `RATE=0.10`
- 8 rows per dataset:
  1. UniMatchV2 official
  2. Base + HPTA
  3. Base + HPTA-MoE
  4. Base + HPTA-MoE + LC-PAM
  5. Base + HPTA-MoE + MGER
  6. Base + HPTA-MoE + SMCR/TMRC
  7. Base + HPTA-MoE + IASL
  8. Full

See:

```bash
docs/SASR_ABLATION_COMMANDS.md
```

## Repository Layout

See:

```bash
docs/REPOSITORY_LAYOUT.md
```

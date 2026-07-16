# Configuration

## Overview

All paper configurations are centralized under `configs/`:

```
configs/
├── paper/
│   └── baseline.yaml           # Single source of truth for paper hyperparameters
├── experiments/
│   ├── personamem.yaml         # PersonaMem experiment
│   ├── locomo.yaml             # LoCoMo experiment
│   └── psydial.yaml            # PsyDial experiment
└── ablations/
    ├── no_consolidation.yaml   # w/o State→Trait Consolidation
    ├── no_branching.yaml       # w/o Context-Conditional Branching
    ├── no_decay.yaml           # w/o Temporal Decay
    ├── uniform_decay.yaml      # w/o Type-Conditioned Decay
    ├── semantic_only.yaml      # Semantic-Only Retrieval
    ├── flat_pool.yaml          # Flat PPMU Pool
    ├── two_level.yaml          # Two-Level Memory
    └── no_evidence_set.yaml    # w/o Evidence Set (post-processing)
```

## Paper Baseline

`configs/paper/baseline.yaml` is the single source of truth for:

| Parameter | Value |
|-----------|-------|
| write_threshold | 0.68 |
| promote_threshold | 0.72 |
| context_threshold | 0.62 |
| promotion_min_sessions | 2 |
| top_k | 5 |

## Experiment Configs

Each experiment config (`configs/experiments/<benchmark>.yaml`):
- References `configs/paper/baseline.yaml`
- Defines only benchmark-specific settings (data paths, phases)
- Does NOT duplicate paper hyperparameters

## Ablation Configs

Each ablation config (`configs/ablations/<variant>.yaml`):
- References `configs/paper/baseline.yaml`
- Overrides only the target mechanism's parameters
- Includes `id`, `description`, `paper_reported`, and `overrides` fields

## Legacy Configs

- `benchmarks/personamem/configs/param_sweep.yaml` — kept for sensitivity analysis experiments; its baseline uses PersonaMem-adjusted weights (NOT the paper baseline)
- `benchmarks/ablations/configs/ablation.yaml` — deprecated monolithic config; replaced by individual files in `configs/ablations/`

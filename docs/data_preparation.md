# Data Preparation

Datasets are NOT distributed with this repository.

## PersonaMem

```bash
# Download from: https://github.com/bowen-upenn/PersonaMem
# Place in: data/datasets/personamem/

ls data/datasets/personamem/
# Expected: shared_contexts_32k.jsonl questions_32k.csv
```

## LoCoMo

```bash
# Download from: https://github.com/snap-research/LoCoMo
# Place in: data/datasets/locomo/

ls data/datasets/locomo/
# Expected: locomo10.json
```

## PsyDial

```bash
# Download from: https://github.com/qiuhuachuan/PsyDial
# Place in: data/datasets/psydial/

ls data/datasets/psydial/
# Expected: PsyDial-D101.json
```

## Legacy Path Mapping

The previous structure stored datasets at:

| Old location | New standard location |
|-------------|----------------------|
| `datasets/PersonaMem/` | `data/datasets/personamem/` |
| `datasets/LoCoMo/` | `data/datasets/locomo/` |
| `datasets/PsyDial/` | `data/datasets/psydial/` |

Benchmark scripts now auto-detect the repo root via `Path(__file__).resolve().parents[3]`.

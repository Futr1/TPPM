# Data Directory

Datasets are NOT distributed with this repository. Download from official sources:

## Download Instructions

| Benchmark | Source | Expected path |
|-----------|--------|--------------|
| PersonaMem | https://github.com/bowen-upenn/PersonaMem | `data/datasets/personamem/` |
| LoCoMo | https://github.com/snap-research/LoCoMo | `data/datasets/locomo/` |
| PsyDial | https://github.com/qiuhuachuan/PsyDial | `data/datasets/psydial/` |

## Migration from Legacy Paths

If your datasets are in the old locations, create symlinks:

```bash
# PersonaMem (old: datasets/PersonaMem/)
ln -s "$(pwd)/datasets/PersonaMem/" data/datasets/personamem

# LoCoMo (old: datasets/LoCoMo/)
ln -s "$(pwd)/datasets/LoCoMo/" data/datasets/locomo

# PsyDial (old: datasets/PsyDial/)
ln -s "$(pwd)/datasets/PsyDial/" data/datasets/psydial
```

See `docs/data_preparation.md` for detailed setup instructions.

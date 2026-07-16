# Third-Party Licenses

This project uses and references third-party components. Each remains subject to its original license.

## Bundled / Modified Code

| Component | Upstream | Local path | License | Redistributed | Notes |
|-----------|----------|-----------|---------|:---:|-------|
| Mini-Agent | [MiniMax](https://github.com/) | `Mini-Agent-5-1/` | MIT | Yes | Core agent framework; TPPM engine is an extension within this framework. See `Mini-Agent-5-1/LICENSE`. |

## Benchmark Datasets

Datasets are NOT distributed with this repository. Download from official sources.

| Dataset | Upstream | Local path (after download) | License | Acquisition |
|---------|----------|---------------------------|---------|-------------|
| PersonaMem | [bowen-upenn/PersonaMem](https://github.com/bowen-upenn/PersonaMem) | `datasets/PersonaMem/` | MIT | [GitHub](https://github.com/bowen-upenn/PersonaMem) |
| LoCoMo | [snap-research/LoCoMo](https://github.com/snap-research/LoCoMo) | `datasets/LoCoMo/` | CC BY-NC 4.0 | [GitHub](https://github.com/snap-research/LoCoMo) |
| PsyDial | [qiuhuachuan/PsyDial](https://github.com/qiuhuachuan/PsyDial) | `datasets/PsyDial/` | Apache 2.0 | [GitHub](https://github.com/qiuhuachuan/PsyDial) |

## Notes

- **LoCoMo** is licensed under CC BY-NC 4.0 (non-commercial). Research use in academic settings is generally permitted; commercial use requires separate permission.
- **PersonaMem** and **PsyDial** use permissive licenses (MIT and Apache 2.0 respectively), but the dialogue content they contain may have ethical considerations regarding redistribution.
- The SFT data generation code in `data_trans/` is exploratory and not part of the formal paper evaluation.
- All dataset references in this repository point to external official sources. Users must comply with each dataset's license terms when downloading and using the data.

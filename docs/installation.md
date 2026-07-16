# Installation

## Prerequisites

- Python 3.10 or later
- Git

## Quick Start

```bash
git clone https://github.com/Futr1/TPPM.git
cd TPPM

python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\Activate.ps1     # Windows PowerShell

pip install -e .
```

## Development Install

```bash
pip install -e ".[dev]"
```

## API Keys

```bash
export DEEPSEEK_API_KEY="your-deepseek-api-key"
```

Optional overrides:

```bash
export DEEPSEEK_API_BASE="https://api.deepseek.com"
export DEEPSEEK_JUDGE_MODEL="deepseek-v4-pro"
```

## Verify

```bash
python3 -c "from tppm.core.memory import TemporalProfileMemory, TPMConfig; print('OK')"
tppm --help
```

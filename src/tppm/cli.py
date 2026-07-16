"""TPPM command-line interface."""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="tppm",
        description="Temporal Psychological Profile Memory — experiment runner",
    )
    sub = parser.add_subparsers(dest="command")

    # describe-datasets
    sub.add_parser("describe-datasets", help="Show dataset download information")

    # run
    run_p = sub.add_parser("run", help="Run a benchmark experiment")
    run_p.add_argument("--config", required=True, help="Path to experiment YAML config")

    # dry-run
    dry_p = sub.add_parser("dry-run", help="Dry-run: validate config without API calls")
    dry_p.add_argument("--config", required=True, help="Path to experiment YAML config")
    dry_p.add_argument("--limit", type=int, default=None, help="Limit to N samples")

    # ablate
    abl_p = sub.add_parser("ablate", help="Run an ablation experiment")
    abl_p.add_argument("--config", required=True, help="Path to ablation YAML config")

    # summarize
    sum_p = sub.add_parser("summarize", help="Summarize run results")
    sum_p.add_argument("--run-dir", required=True, help="Path to run directory")

    args = parser.parse_args()

    if args.command == "describe-datasets":
        _cmd_describe_datasets()
    elif args.command == "dry-run":
        _cmd_dry_run(args)
    elif args.command == "run":
        print(f"Run: {args.config}")
        print("(experiment runner — use benchmark scripts directly for now)")
    elif args.command == "ablate":
        print(f"Ablate: {args.config}")
        print("(ablation runner — use ablation scripts directly for now)")
    elif args.command == "summarize":
        print(f"Summarize: {args.run_dir}")
    else:
        parser.print_help()


def _cmd_describe_datasets() -> None:
    print("Datasets are not distributed with TPPM. Download from official sources:\n")
    print("  PersonaMem  → https://github.com/bowen-upenn/PersonaMem")
    print("    Place under: data/datasets/personamem/")
    print("    Files: shared_contexts_32k.jsonl, questions_32k.csv\n")
    print("  LoCoMo      → https://github.com/snap-research/LoCoMo")
    print("    Place under: data/datasets/locomo/")
    print("    Files: locomo10.json\n")
    print("  PsyDial     → https://github.com/qiuhuachuan/PsyDial")
    print("    Place under: data/datasets/psydial/")
    print("    Files: PsyDial-D101.json\n")
    print("See data/README.md and docs/data_preparation.md for details.")


def _cmd_dry_run(args: argparse.Namespace) -> None:
    from pathlib import Path

    import yaml

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: config not found: {args.config}")
        sys.exit(1)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    print(f"Benchmark:    {config.get('benchmark', 'unknown')}")
    print(f"Config:       {args.config}")
    print(f"Limit:        {args.limit or 'full'}")
    print(f"Data paths:   {config.get('data', {})}")
    if "phases" in config:
        print(f"Phases:       {list(config['phases'].keys())}")
    print(f"Output dir:   {config.get('output', {}).get('run_dir', 'runs/')}")
    print("\nNo API calls made. Ready to run.")


if __name__ == "__main__":
    main()

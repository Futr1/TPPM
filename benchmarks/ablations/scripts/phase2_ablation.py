#!/usr/bin/env python3
"""Phase 2 benchmarks/ablations: Replay memory evolution for each ablation variant.

Loads Phase 1 candidates from benchmarks/personamem, replays all sessions through
TemporalProfileMemory for each ablation config. Pure Python — zero LLM calls.

Usage:
    python3 phase2_ablation.py                                    # all ablation configs
    python3 phase2_ablation.py --config-id ablation_consolidation  # single config
    python3 phase2_ablation.py --dry-run                           # list configs only
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
from typing import Any

# Allow importing Mini-Agent-5-1 TPPM modules

from tppm.core.memory import TemporalProfileMemory, TPMConfig
from tppm.core.models import ProfileCandidate

import yaml
from tqdm import tqdm

# ===== Paths =====
ROOT = REPO_ROOT / 'benchmarks/ablations'
TABLE3_DATA = REPO_ROOT / 'benchmarks/personamem'
CANDIDATES_DIR = TABLE3_DATA / "candidates"
SNAPSHOTS_DIR = ROOT / "memory_snapshots"
ABLATION_CONFIG_PATH = ROOT / "configs" / "ablation.yaml"

# ===== Simulated time between sessions (for long-term decay) =====
SESSION_INTERVAL_HOURS = 24

# ===== Config IDs that need special handling in replay =====
NO_DECAY_CONFIGS = {"ablation_decay"}

# ===== Config loading =====

def load_ablation_configs() -> dict[str, dict[str, Any]]:
    """Load ablation config definitions from YAML."""
    with ABLATION_CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f.read())

def _make_config(params: dict[str, Any]) -> TPMConfig:
    """Create TPMConfig from parameter dict."""
    decay_lambdas = dict(params.get("decay_lambdas", {
        "goal": 0.1, "interest": 0.07, "style": 0.04,
        "background": 0.04, "preference": 0.05, "general": 0.05,
    }))
    return TPMConfig(
        write_threshold=float(params.get("write_threshold", 0.68)),
        promote_threshold=float(params.get("promote_threshold", 0.58)),
        context_threshold=float(params.get("context_threshold", 0.62)),
        decay_lambdas=decay_lambdas,
        promote_weights=tuple(params.get("promote_weights", (0.10, 0.10, 0.05, 0.70, 0.05))),
        promotion_min_sessions=int(params.get("promotion_min_sessions", 1)),
    )

def resolve_configs(
    configs_data: dict[str, Any],
    config_id: str | None = None,
) -> list[tuple[str, TPMConfig]]:
    """Resolve ablation configs into (config_id, TPMConfig) pairs."""
    configs: list[tuple[str, TPMConfig]] = []
    for cid, params in configs_data.items():
        if not isinstance(params, dict):
            continue
        if config_id and cid != config_id:
            continue
        configs.append((cid, _make_config(params)))
    return configs

# ===== Candidate loading =====

def load_context_candidates(context_dir: Path) -> list[tuple[int, list[dict[str, Any]]]]:
    """Load all session candidate files for a context, sorted by session_idx."""
    sessions: list[tuple[int, list[dict[str, Any]]]] = []
    for fpath in sorted(context_dir.glob("session_*.json")):
        with fpath.open("r", encoding="utf-8") as f:
            data = json.load(f)
        sessions.append((data["session_idx"], data.get("candidates", [])))
    sessions.sort(key=lambda x: x[0])
    return sessions

def candidates_to_objects(raw_list: list[dict[str, Any]]) -> list[ProfileCandidate]:
    """Convert candidate dicts to ProfileCandidate objects."""
    objs: list[ProfileCandidate] = []
    for item in raw_list:
        try:
            objs.append(ProfileCandidate(
                attribute=item["attribute"],
                value=item["value"],
                context=item.get("context", ""),
                profile_type=item.get("profile_type", "general"),
                scene=item.get("scene", "general"),
                confidence=float(item.get("confidence", 0.7)),
                stability=float(item.get("stability", 0.5)),
                recency=float(item.get("recency", 1.0)),
                explicitness=float(item.get("explicitness", 0.7)),
                user_relevance=float(item.get("user_relevance", 0.75)),
                source=item.get("source", "llm_deepseek"),
            ))
        except (KeyError, TypeError, ValueError) as e:
            tqdm.write(f"[WARN] Skipping malformed candidate: {e}")
            continue
    return objs

# ===== Replay engine =====

def replay_context(
    context_hash: str,
    config: TPMConfig,
    config_id: str,
) -> dict[str, Any] | None:
    """Replay all sessions for one context through TPPM with given config.

    For ablation_decay: skips decay_long_term() to disable temporal decay.
    """
    context_dir = CANDIDATES_DIR / context_hash
    if not context_dir.exists():
        tqdm.write(f"[WARN] No candidates for context {context_hash[:8]}")
        return None

    sessions = load_context_candidates(context_dir)
    if not sessions:
        return None

    tpm = TemporalProfileMemory(config)
    skip_ltm_decay = (config_id in NO_DECAY_CONFIGS)

    for session_idx, raw_candidates in sessions:
        scene = f"session_{session_idx}"
        session_id = f"{context_hash}_session_{session_idx}"

        tpm.start_session(scene=scene, session_id=session_id)

        candidates = candidates_to_objects(raw_candidates)
        if candidates:
            ingested = tpm.ingest_candidates(candidates, scene=scene, session_id=session_id)
            for cand in candidates:
                tpm.retrieve(cand.attribute, scene=scene, top_k=3)

        tpm.finish_session(scene=scene)
        tpm.run_evolution_engine(scene=scene, include_long_term_decay=False)

        # Simulate time passage for decay calculation
        for pmu in tpm.long_term_memory:
            try:
                last = datetime.fromisoformat(pmu.last_evolved)
            except (ValueError, TypeError):
                last = datetime.utcnow()
            pmu.last_evolved = (last - timedelta(hours=SESSION_INTERVAL_HOURS)).isoformat()

        # --- ABLATION: skip long-term decay ---
        if not skip_ltm_decay:
            tpm.decay_long_term()

    snapshot = tpm.to_dict()
    snapshot["config_id"] = config_id
    snapshot["context_hash"] = context_hash
    snapshot["num_sessions"] = len(sessions)
    return snapshot

def run_replay(configs: list[tuple[str, TPMConfig]]) -> dict[str, int]:
    """Run memory replay for all configs across all contexts."""
    if not CANDIDATES_DIR.exists():
        print("[ERROR] No candidates directory. Run Phase 1 first.")
        return {}

    context_hashes = sorted(
        d.name for d in CANDIDATES_DIR.iterdir()
        if d.is_dir() and (d / "session_000.json").exists()
    )
    if not context_hashes:
        print("[ERROR] No candidate directories found.")
        return {}

    print(f"[INFO] Contexts: {len(context_hashes)}")
    print(f"[INFO] Configs: {len(configs)}")

    stats: dict[str, int] = {}

    for config_id, config in tqdm(configs, desc="Configs"):
        output_dir = SNAPSHOTS_DIR / config_id
        output_dir.mkdir(parents=True, exist_ok=True)

        n_processed = 0
        for ctx_hash in tqdm(context_hashes, desc=f"  {config_id}", leave=False):
            snapshot = replay_context(ctx_hash, config, config_id)
            if snapshot is None:
                continue
            output_path = output_dir / f"{ctx_hash}.json"
            with output_path.open("w", encoding="utf-8") as f:
                json.dump(snapshot, f, ensure_ascii=False, indent=2)
            n_processed += 1

        stats[config_id] = n_processed

    return stats

# ===== CLI =====

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Phase 2 benchmarks/ablations: Replay memory evolution for ablation configs")
    parser.add_argument("--config-id", type=str, default=None,
                        help="Single ablation config ID (e.g. ablation_consolidation)")
    parser.add_argument("--include-baseline", action="store_true",
                        help="Also regenerate baseline (normally skipped — reuses existing)")
    parser.add_argument("--dry-run", action="store_true",
                        help="List configs without running")
    args = parser.parse_args()

    configs_data = load_ablation_configs()

    # By default, skip baseline (already exists in benchmarks/personamem)
    if not args.include_baseline:
        configs_data.pop("baseline", None)

    configs = resolve_configs(configs_data, config_id=args.config_id)

    print(f"[INFO] Resolved {len(configs)} configs:")
    for cid, cfg in configs:
        print(f"  {cid}: write_thr={cfg.write_threshold}, "
              f"promote_thr={cfg.promote_threshold}, "
              f"ctx_thr={cfg.context_threshold}, "
              f"decay_goal={cfg.decay_lambdas.get('goal', 0.1)}, "
              f"no_ltm_decay={'YES' if cid in NO_DECAY_CONFIGS else 'no'}")

    if args.dry_run:
        return 0

    stats = run_replay(configs)

    print(f"\n[DONE] Phase 2 benchmarks/ablations — {len(stats)} configs:")
    for cid, n in stats.items():
        print(f"  {cid}: {n} contexts")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())

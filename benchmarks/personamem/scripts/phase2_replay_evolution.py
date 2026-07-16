#!/usr/bin/env python3
"""Phase 2: Replay memory evolution for each parameter config.

Loads Phase 1 candidates, replays all sessions through TemporalProfileMemory
for each parameter config. Pure Python — zero LLM calls.

Usage:
    python3 phase2_replay_evolution.py                           # all configs
    python3 phase2_replay_evolution.py --config-id write_0.56    # single config
    python3 phase2_replay_evolution.py --sweep sweep_2a_write     # one sweep group
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
_AGENT_ROOT = Path("/root/autodl-tmp/wangqihao/Mini-Agent-5-1")
if str(_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(_AGENT_ROOT))

from mini_agent.tpm.memory import TemporalProfileMemory, TPMConfig
from mini_agent.tpm.models import ProfileCandidate

import yaml
from tqdm import tqdm

# ===== Paths =====
ROOT = REPO_ROOT / 'benchmarks/personamem'
CANDIDATES_DIR = ROOT / "candidates"
SNAPSHOTS_DIR = ROOT / "memory_snapshots"
SWEEP_CONFIG_PATH = ROOT / "configs" / "param_sweep.yaml"

# ===== Simulated time between sessions (for long-term decay) =====
SESSION_INTERVAL_HOURS = 24  # Simulate 1 day between sessions


# ===== Config loading =====

def load_sweep_configs(sweep_path: Path) -> dict[str, Any]:
    """Load parameter sweep definitions from YAML."""
    with sweep_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f.read())


def resolve_configs(
    sweep_data: dict[str, Any],
    sweep_name: str | None = None,
    config_id: str | None = None,
) -> list[tuple[str, TPMConfig]]:
    """Resolve sweep definitions into (config_id, TPMConfig) pairs.

    Args:
        sweep_data: Parsed YAML data.
        sweep_name: If given, only process this sweep group (e.g. 'sweep_2a_write').
        config_id: If given, only process this single config.

    Returns:
        List of (config_id, TPMConfig) tuples.
    """
    baseline = sweep_data.get("baseline", {})
    configs: list[tuple[str, TPMConfig]] = []

    # Always include baseline
    configs.append(("baseline", _make_config(baseline)))

    # Process each sweep group
    for key, sweep_def in sweep_data.items():
        if key == "baseline" or not isinstance(sweep_def, dict):
            continue

        if sweep_name and key != sweep_name:
            continue

        for cfg in sweep_def.get("configs", []):
            cid = cfg["config_id"]
            if config_id and cid != config_id:
                continue

            # Merge with baseline
            merged = {**baseline}
            for k, v in cfg.items():
                if k not in ("config_id",):
                    merged[k] = v

            configs.append((cid, _make_config(merged)))

    return configs


def _make_config(params: dict[str, Any]) -> TPMConfig:
    """Create TPMConfig from parameter dict, handling decay_lambdas scaling."""
    decay_lambdas = dict(params.get("decay_lambdas", {
        "goal": 0.1, "interest": 0.07, "style": 0.04,
        "background": 0.04, "preference": 0.05, "general": 0.05,
    }))

    # Apply global scale factor if present
    scale = params.get("decay_lambdas_scale", 1.0)
    if scale != 1.0:
        decay_lambdas = {k: v * scale for k, v in decay_lambdas.items()}

    return TPMConfig(
        write_threshold=float(params.get("write_threshold", 0.68)),
        promote_threshold=float(params.get("promote_threshold", 0.58)),
        context_threshold=float(params.get("context_threshold", 0.62)),
        decay_lambdas=decay_lambdas,
        promote_weights=tuple(params.get("promote_weights", (0.10, 0.10, 0.05, 0.70, 0.05))),
        promotion_min_sessions=1,  # PersonaMem: each context = one persona, sessions are segments
    )


# ===== Candidate loading =====

def load_context_candidates(context_dir: Path) -> list[tuple[int, list[dict[str, Any]]]]:
    """Load all session candidate files for a context, sorted by session_idx.

    Returns:
        list of (session_idx, candidate_dicts) sorted chronologically.
    """
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

    Returns:
        Memory snapshot dict, or None if no candidates found.
    """
    context_dir = CANDIDATES_DIR / context_hash
    if not context_dir.exists():
        tqdm.write(f"[WARN] No candidates for context {context_hash[:8]}")
        return None

    sessions = load_context_candidates(context_dir)
    if not sessions:
        return None

    tpm = TemporalProfileMemory(config)

    for session_idx, raw_candidates in sessions:
        scene = f"session_{session_idx}"
        session_id = f"{context_hash}_session_{session_idx}"

        tpm.start_session(scene=scene, session_id=session_id)

        candidates = candidates_to_objects(raw_candidates)
        if candidates:
            ingested = tpm.ingest_candidates(candidates, scene=scene, session_id=session_id)

            # Simulate conversation-time retrieval: query TPPM with each
            # candidate's attribute to increment access_count on matched PMUs.
            # This is what a real agent would do during dialogue to recall
            # relevant profile memories.
            for cand in candidates:
                tpm.retrieve(cand.attribute, scene=scene, top_k=3)

        tpm.finish_session(scene=scene)

        # Run evolution engine after each session:
        # - Integrate scene views (refresh canonical view per branch)
        # - Decay working memory (0.015) and short-term memory (0.03)
        tpm.run_evolution_engine(scene=scene, include_long_term_decay=False)

        # Simulate time passage: backdate LTM last_evolved so decay_long_term
        # sees a meaningful Δt. PersonaMem sessions are synthetic — without this,
        # Δt≈0 and decay has no effect.
        for pmu in tpm.long_term_memory:
            try:
                last = datetime.fromisoformat(pmu.last_evolved)
            except (ValueError, TypeError):
                last = datetime.utcnow()
            pmu.last_evolved = (last - timedelta(hours=SESSION_INTERVAL_HOURS)).isoformat()

        tpm.decay_long_term()

    snapshot = tpm.to_dict()
    snapshot["config_id"] = config_id
    snapshot["context_hash"] = context_hash
    snapshot["num_sessions"] = len(sessions)
    return snapshot


def run_replay(
    configs: list[tuple[str, TPMConfig]],
) -> dict[str, int]:
    """Run memory replay for all configs across all contexts.

    Returns:
        dict mapping config_id to number of contexts processed.
    """
    # Discover all contexts from Phase 1 output
    if not CANDIDATES_DIR.exists():
        print("[ERROR] No candidates directory. Run Phase 1 first.")
        return {}

    context_hashes = sorted(
        d.name for d in CANDIDATES_DIR.iterdir()
        if d.is_dir() and (d / "session_000.json").exists()
    )
    if not context_hashes:
        print("[ERROR] No candidate directories found. Run Phase 1 first.")
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
        description="Phase 2: Replay memory evolution for parameter sweeps")
    parser.add_argument("--sweep", type=str, default=None,
                        help="Sweep group name (e.g. sweep_2a_write)")
    parser.add_argument("--config-id", type=str, default=None,
                        help="Single config ID (e.g. write_0.56)")
    parser.add_argument("--dry-run", action="store_true",
                        help="List configs without running")
    args = parser.parse_args()

    sweep_data = load_sweep_configs(SWEEP_CONFIG_PATH)
    configs = resolve_configs(sweep_data, sweep_name=args.sweep, config_id=args.config_id)

    print(f"[INFO] Resolved {len(configs)} configs:")
    for cid, cfg in configs:
        decay_scale = cfg.decay_lambdas.get("goal", 0.1) / 0.1 if cfg.decay_lambdas else 1.0
        print(f"  {cid}: write_thr={cfg.write_threshold}, promote_thr={cfg.promote_threshold}, "
              f"ctx_thr={cfg.context_threshold}, decay_scale≈{decay_scale:.2f}x")

    if args.dry_run:
        return 0

    stats = run_replay(configs)

    print(f"\n[DONE] Processed {len(stats)} configs:")
    for cid, n in stats.items():
        print(f"  {cid}: {n} contexts")
    print(f"[DONE] Output: {SNAPSHOTS_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

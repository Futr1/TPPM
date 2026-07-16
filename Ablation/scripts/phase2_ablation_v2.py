#!/usr/bin/env python3
"""Phase 2 Ablation v2: Replay memory evolution for fine-grained & hierarchy ablation variants.

Supports three architecture modes:
  - "standard": Normal 3-tier TPPM (baseline + existing ablations)
  - "flat": Single pool for all PMUs, no tier distinction
  - "two_level": Merge working+short-term into transient, keep long-term

New variants:
  - ablation_uniform_decay: Type-conditioned decay → uniform decay rate
  - ablation_semantic_retrieval: No re-extraction needed (flag: reuse_baseline=true)
  - ablation_flat_pool: Flat PPMU pool (architecture=flat)
  - ablation_two_level: Two-level memory (architecture=two_level)

Usage:
    python3 phase2_ablation_v2.py                                    # all new variants
    python3 phase2_ablation_v2.py --config-id ablation_uniform_decay  # single variant
    python3 phase2_ablation_v2.py --dry-run                           # list configs only
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timedelta
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

# Allow importing Mini-Agent-5-1 TPPM modules
_AGENT_ROOT = Path("/root/autodl-tmp/wangqihao/Mini-Agent-5-1")
if str(_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(_AGENT_ROOT))

from mini_agent.tpm.memory import TemporalProfileMemory, TPMConfig, _clamp, _parse_timestamp
from mini_agent.tpm.models import ProfileCandidate, ProfileMemoryUnit, EvidenceItem, utc_now

import yaml
from tqdm import tqdm

# ===== Paths =====
ROOT = Path("/root/autodl-tmp/wangqihao/Ablation")
TABLE3_DATA = Path("/root/autodl-tmp/wangqihao/Table3-data")
CANDIDATES_DIR = TABLE3_DATA / "candidates"
SNAPSHOTS_DIR = ROOT / "memory_snapshots"
ABLATION_CONFIG_PATH = ROOT / "configs" / "ablation.yaml"

# ===== Simulated time between sessions =====
SESSION_INTERVAL_HOURS = 24

# ===== New variant config IDs =====
SKIP_EXTRACTION_CONFIGS = {"ablation_semantic_retrieval"}  # reuse baseline


# ===== Helper functions (copied from phase2_ablation.py for independence) =====

def _normalize(text: str) -> str:
    return " ".join((text or "").lower().split())


def _similarity(left: str, right: str) -> float:
    left_norm = _normalize(left)
    right_norm = _normalize(right)
    if not left_norm or not right_norm:
        return 0.0
    return SequenceMatcher(None, left_norm, right_norm).ratio()


# ===== Config loading =====

def load_ablation_configs() -> dict[str, dict[str, Any]]:
    with ABLATION_CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f.read())


def _make_config(params: dict[str, Any]) -> TPMConfig:
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


# ===== Candidate loading =====

def load_context_candidates(context_dir: Path) -> list[tuple[int, list[dict[str, Any]]]]:
    sessions: list[tuple[int, list[dict[str, Any]]]] = []
    for fpath in sorted(context_dir.glob("session_*.json")):
        with fpath.open("r", encoding="utf-8") as f:
            data = json.load(f)
        sessions.append((data["session_idx"], data.get("candidates", [])))
    sessions.sort(key=lambda x: x[0])
    return sessions


def candidates_to_objects(raw_list: list[dict[str, Any]]) -> list[ProfileCandidate]:
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


# ===== Flat PPMU Pool Architecture =====

class FlatTPPM:
    """Single-pool TPPM: all PMUs in one store, no tier distinction."""

    def __init__(self, config: TPMConfig | None = None):
        self.config = config or TPMConfig()
        self.pool: list[ProfileMemoryUnit] = []  # single unified pool
        self.evidence_store: dict[str, EvidenceItem] = {}
        self.current_session: str | None = None
        self.current_session_id: str | None = None

    def start_session(self, scene: str = "general", session_id: str | None = None) -> None:
        self.current_session = scene or "general"
        self.current_session_id = session_id or self.current_session_id

    def ingest_candidates(
        self,
        candidates: list[ProfileCandidate],
        scene: str = "general",
        session_id: str | None = None,
    ) -> list[ProfileMemoryUnit]:
        if self.current_session is None or self.current_session_id != session_id:
            self.start_session(scene, session_id=session_id)
        else:
            self.current_session = scene or "general"

        accepted: list[ProfileMemoryUnit] = []
        for candidate in candidates:
            if candidate.write_score(self.config.write_weights) < self.config.write_threshold:
                continue
            accepted.append(self._align_or_create(candidate, session_id=session_id))
        return accepted

    def finish_session(self, scene: str | None = None) -> None:
        """In flat pool, there's no tier transition — just keep all in pool."""
        scene = scene or self.current_session or "general"
        # No working→short-term transition needed; everything is already in pool
        # Still run promotion logic as "quality scoring" but don't move between tiers
        self.current_session = scene

    def run_evolution_engine(self, scene: str = "general", *, include_long_term_decay: bool = False) -> None:
        """Run evolution on the unified pool with a single moderate decay rate."""
        self._integrate_scene_views(scene)
        # Apply uniform decay to all PMUs in the pool
        uniform_decay = 0.03  # moderate decay rate for flat pool
        self._decay_store(self.pool, uniform_decay)

    def _integrate_scene_views(self, scene: str) -> None:
        for unit in self.pool:
            unit.scene_view(scene)
            unit.refresh_canonical_view()

    def _decay_store(self, store: list[ProfileMemoryUnit], decay_rate: float) -> None:
        if decay_rate <= 0:
            return
        now = utc_now()
        for unit in store:
            delta_hours = max((now - _parse_timestamp(unit.last_evolved)).total_seconds() / 3600.0, 0.0)
            if delta_hours <= 0:
                continue
            # Use uniform decay (ignore profile_type)
            unit.stability_score = _clamp(unit.stability_score * math.exp(-decay_rate * delta_hours / 24.0))
            unit.last_evolved = now.isoformat()

    def _align_or_create(self, candidate: ProfileCandidate, session_id: str | None = None) -> ProfileMemoryUnit:
        matched = self._best_match(candidate)
        evidence = EvidenceItem(
            source=candidate.source,
            content=candidate.context,
            scene=candidate.scene,
            timestamp=candidate.timestamp,
        )
        self.evidence_store[evidence.evidence_id] = evidence

        if matched is None:
            unit = ProfileMemoryUnit(
                attribute=candidate.attribute,
                value=candidate.value,
                context=candidate.context,
                profile_type=candidate.profile_type,
                stability_score=candidate.stability,
                confidence_score=candidate.confidence,
                scene=candidate.scene,
                quality_score=candidate.quality_score,
                evidence=[evidence],
                memory_level="pool",  # flat pool — all same level
                session_count=1,
                reinforcement_count=1,
                contradiction_count=0,
                access_count=0,
                seen_session_ids=[session_id] if session_id else [],
            )
            unit.ensure_branch(
                candidate.scene,
                value=candidate.value,
                context=candidate.context,
                confidence_score=candidate.confidence,
                quality_score=candidate.quality_score,
            ).add_evidence(evidence)
            self.pool.append(unit)
            return unit

        self._fuse_candidate(matched, candidate, evidence, session_id=session_id)
        return matched

    def _best_match(self, candidate: ProfileCandidate) -> ProfileMemoryUnit | None:
        best_score = 0.0
        best_unit: ProfileMemoryUnit | None = None
        for unit in self.pool:
            score = self._similarity(candidate, unit)
            if score > best_score:
                best_score = score
                best_unit = unit
        if best_score >= self.config.context_threshold:
            return best_unit
        return None

    def _similarity(self, candidate: ProfileCandidate, unit: ProfileMemoryUnit) -> float:
        branch = unit.scene_view(candidate.scene)
        attr_score = 1.0 if candidate.attribute == unit.attribute else 0.0
        type_score = 1.0 if candidate.profile_type == unit.profile_type else 0.35
        value_score = max(_similarity(candidate.value, branch.value), _similarity(candidate.value, unit.value))
        context_score = max(_similarity(candidate.context, branch.context), _similarity(candidate.context, unit.context))
        if candidate.scene == branch.scene:
            scene_score = 1.0
        elif candidate.scene == "general" or branch.scene == "general":
            scene_score = 0.65
        else:
            scene_score = 0.35
        return 0.25 * attr_score + 0.25 * value_score + 0.2 * context_score + 0.1 * type_score + 0.2 * scene_score

    def _fuse_candidate(
        self,
        unit: ProfileMemoryUnit,
        candidate: ProfileCandidate,
        evidence: EvidenceItem,
        *,
        session_id: str | None,
    ) -> None:
        branch = unit.ensure_branch(
            candidate.scene,
            value=candidate.value,
            context=candidate.context,
            confidence_score=candidate.confidence,
            quality_score=candidate.quality_score,
        )
        contradiction = bool(
            candidate.value
            and branch.value
            and _similarity(candidate.value, branch.value) < 0.35
            and candidate.attribute == unit.attribute
        )
        if contradiction:
            unit.contradiction_count += 1
            branch.contradiction_count += 1

        branch.reinforcement_count += 1
        if not contradiction or candidate.confidence >= branch.confidence_score:
            branch.value = candidate.value
        branch.context = candidate.context or branch.context
        branch.confidence_score = _clamp(0.6 * branch.confidence_score + 0.4 * candidate.confidence)
        branch.quality_score = _clamp(0.55 * branch.quality_score + 0.45 * candidate.quality_score)
        branch.add_evidence(evidence)

        unit.add_evidence(evidence)
        unit.reinforcement_count += 1
        if session_id and session_id not in unit.seen_session_ids:
            unit.seen_session_ids.append(session_id)
        unit.session_count = max(1, len(unit.seen_session_ids) or unit.session_count)
        unit.confidence_score = _clamp(0.6 * unit.confidence_score + 0.4 * candidate.confidence)
        unit.quality_score = _clamp(0.55 * unit.quality_score + 0.45 * candidate.quality_score)
        observed_stability = candidate.stability + self.config.positive_reinforcement * min(
            1.0, branch.reinforcement_count / 3.0
        )
        if contradiction:
            observed_stability -= self.config.negative_penalty * 0.5
        unit.stability_score = _clamp(0.6 * unit.stability_score + 0.4 * observed_stability)
        unit.scene = candidate.scene
        unit.last_evolved = evidence.timestamp
        unit.refresh_canonical_view()

    def retrieve(self, query: str, scene: str = "general", top_k: int = 5) -> list[ProfileMemoryUnit]:
        """Retrieve from the unified pool (same scoring as standard TPM)."""
        query_norm = _normalize(query)
        scored: list[tuple[float, ProfileMemoryUnit]] = []
        seen_unit_ids: set[str] = set()
        for unit in self.pool:
            if unit.unit_id in seen_unit_ids:
                continue
            branch = unit.scene_view(scene)
            rel = max(_similarity(query_norm, branch.value), _similarity(query_norm, unit.value))
            scene_score = 1.0 if branch.scene == scene else 0.7 if branch.scene == "general" or scene == "general" else 0.4
            ctx_score = max(
                _similarity(query_norm, branch.context),
                _similarity(query_norm, unit.context),
                1.0 if unit.attribute in query_norm or unit.profile_type in query_norm else 0.0,
            )
            w1, w2, w3, w4, w5 = self.config.retrieve_weights
            score = (
                w1 * rel
                + w2 * unit.stability_score
                + w3 * ctx_score
                + w4 * scene_score
                + w5 * max(unit.quality_score, branch.quality_score)
            )
            if score <= 0:
                continue
            unit.touch_access(scene)
            scored.append((score, unit))
            seen_unit_ids.add(unit.unit_id)
        scored.sort(key=lambda item: item[0], reverse=True)
        return [unit for _, unit in scored[:top_k]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": {
                "write_threshold": self.config.write_threshold,
                "context_threshold": self.config.context_threshold,
                "promote_threshold": self.config.promote_threshold,
                "promotion_min_sessions": self.config.promotion_min_sessions,
                "decay_lambdas": self.config.decay_lambdas,
                "positive_reinforcement": self.config.positive_reinforcement,
                "negative_penalty": self.config.negative_penalty,
            },
            # Store all PMUs under "long_term_memory" for compatibility with eval scripts
            "working_memory": [],
            "short_term_memory": [],
            "long_term_memory": [unit.to_dict() for unit in self.pool],
            "evidence_store": {
                eid: item.to_dict() for eid, item in self.evidence_store.items()
            },
            "current_session": self.current_session,
            "current_session_id": self.current_session_id,
            "architecture": "flat",
        }


# ===== Two-Level Memory Architecture =====

class TwoLevelTPPM:
    """Two-level TPM: transient (working+short-term merged) + long-term."""

    def __init__(self, config: TPMConfig | None = None):
        self.config = config or TPMConfig()
        self.transient_memory: list[ProfileMemoryUnit] = []  # merged working+short-term
        self.long_term_memory: list[ProfileMemoryUnit] = []
        self.evidence_store: dict[str, EvidenceItem] = {}
        self.current_session: str | None = None
        self.current_session_id: str | None = None

    def start_session(self, scene: str = "general", session_id: str | None = None) -> None:
        self.current_session = scene or "general"
        self.current_session_id = session_id or self.current_session_id

    def ingest_candidates(
        self,
        candidates: list[ProfileCandidate],
        scene: str = "general",
        session_id: str | None = None,
    ) -> list[ProfileMemoryUnit]:
        if self.current_session is None or self.current_session_id != session_id:
            self.start_session(scene, session_id=session_id)
        else:
            self.current_session = scene or "general"

        accepted: list[ProfileMemoryUnit] = []
        for candidate in candidates:
            if candidate.write_score(self.config.write_weights) < self.config.write_threshold:
                continue
            accepted.append(self._align_or_create(candidate, session_id=session_id))
        return accepted

    def finish_session(self, scene: str | None = None) -> None:
        """In two-level: all PMUs stay in transient; then promote stable ones to long-term."""
        scene = scene or self.current_session or "general"
        for unit in list(self.transient_memory):
            if unit.memory_level == "working":
                unit.memory_level = "transient"
                unit.scene = scene
                unit.refresh_canonical_view()

        self._promote_stable_memories()
        self.current_session = scene

    def run_evolution_engine(self, scene: str = "general", *, include_long_term_decay: bool = False) -> None:
        self._integrate_scene_views(scene)
        # Transient uses combined decay (average of working + short-term)
        transient_decay = (self.config.working_decay + self.config.short_term_decay) / 2.0
        self._decay_store(self.transient_memory, transient_decay)
        if include_long_term_decay:
            self.decay_long_term()

    def decay_long_term(self) -> None:
        now = utc_now()
        for unit in self.long_term_memory:
            delta_hours = max((now - _parse_timestamp(unit.last_evolved)).total_seconds() / 3600.0, 0.0)
            if delta_hours <= 0:
                continue
            decay = self.config.decay_lambdas.get(unit.profile_type, self.config.decay_lambdas["general"])
            positive_signal = min(1.0, unit.reinforcement_count / max(1, unit.session_count * 2))
            negative_signal = min(1.0, unit.contradiction_count / max(1, unit.reinforcement_count))
            unit.stability_score = _clamp(
                unit.stability_score * math.exp(-decay * delta_hours / 24.0)
                + self.config.positive_reinforcement * positive_signal
                - self.config.negative_penalty * negative_signal
            )
            unit.last_evolved = now.isoformat()

    def _integrate_scene_views(self, scene: str) -> None:
        for unit in [*self.transient_memory, *self.long_term_memory]:
            unit.scene_view(scene)
            unit.refresh_canonical_view()

    def _decay_store(self, store: list[ProfileMemoryUnit], decay_rate: float) -> None:
        if decay_rate <= 0:
            return
        now = utc_now()
        for unit in store:
            delta_hours = max((now - _parse_timestamp(unit.last_evolved)).total_seconds() / 3600.0, 0.0)
            if delta_hours <= 0:
                continue
            unit.stability_score = _clamp(unit.stability_score * math.exp(-decay_rate * delta_hours / 24.0))
            unit.last_evolved = now.isoformat()

    def _promote_stable_memories(self) -> None:
        """Promote transient PMUs that meet long-term criteria."""
        promote_weights = self.config.promote_weights
        kept_transient: list[ProfileMemoryUnit] = []
        for unit in self.transient_memory:
            evidence_strength = min(1.0, len(unit.evidence) / 4.0)
            usage_strength = min(1.0, unit.access_count / 3.0)
            reinforcement_strength = min(1.0, unit.reinforcement_count / 4.0)
            contradiction_strength = min(1.0, unit.contradiction_count / 3.0)
            score = (
                promote_weights[0] * reinforcement_strength
                + promote_weights[1] * evidence_strength
                + promote_weights[2] * usage_strength
                + promote_weights[3] * unit.stability_score
                - promote_weights[4] * contradiction_strength
            )
            if score >= self.config.promote_threshold and unit.session_count >= self.config.promotion_min_sessions:
                unit.memory_level = "long_term"
                self._merge_into_store(self.long_term_memory, unit)
            else:
                kept_transient.append(unit)
        self.transient_memory = kept_transient

    def _align_or_create(self, candidate: ProfileCandidate, session_id: str | None = None) -> ProfileMemoryUnit:
        matched = self._best_match(candidate)
        evidence = EvidenceItem(
            source=candidate.source,
            content=candidate.context,
            scene=candidate.scene,
            timestamp=candidate.timestamp,
        )
        self.evidence_store[evidence.evidence_id] = evidence

        if matched is None:
            unit = ProfileMemoryUnit(
                attribute=candidate.attribute,
                value=candidate.value,
                context=candidate.context,
                profile_type=candidate.profile_type,
                stability_score=candidate.stability,
                confidence_score=candidate.confidence,
                scene=candidate.scene,
                quality_score=candidate.quality_score,
                evidence=[evidence],
                memory_level="working",  # starts as working in transient
                session_count=1,
                reinforcement_count=1,
                contradiction_count=0,
                access_count=0,
                seen_session_ids=[session_id] if session_id else [],
            )
            unit.ensure_branch(
                candidate.scene,
                value=candidate.value,
                context=candidate.context,
                confidence_score=candidate.confidence,
                quality_score=candidate.quality_score,
            ).add_evidence(evidence)
            self.transient_memory.append(unit)
            return unit

        self._fuse_candidate(matched, candidate, evidence, session_id=session_id)
        return matched

    def _best_match(self, candidate: ProfileCandidate) -> ProfileMemoryUnit | None:
        stores = [*self.transient_memory, *self.long_term_memory]
        best_score = 0.0
        best_unit: ProfileMemoryUnit | None = None
        for unit in stores:
            score = self._similarity(candidate, unit)
            if score > best_score:
                best_score = score
                best_unit = unit
        if best_score >= self.config.context_threshold:
            return best_unit
        return None

    def _similarity(self, candidate: ProfileCandidate, unit: ProfileMemoryUnit) -> float:
        branch = unit.scene_view(candidate.scene)
        attr_score = 1.0 if candidate.attribute == unit.attribute else 0.0
        type_score = 1.0 if candidate.profile_type == unit.profile_type else 0.35
        value_score = max(_similarity(candidate.value, branch.value), _similarity(candidate.value, unit.value))
        context_score = max(_similarity(candidate.context, branch.context), _similarity(candidate.context, unit.context))
        if candidate.scene == branch.scene:
            scene_score = 1.0
        elif candidate.scene == "general" or branch.scene == "general":
            scene_score = 0.65
        else:
            scene_score = 0.35
        return 0.25 * attr_score + 0.25 * value_score + 0.2 * context_score + 0.1 * type_score + 0.2 * scene_score

    def _merge_into_store(self, store: list[ProfileMemoryUnit], unit: ProfileMemoryUnit) -> None:
        for existing in store:
            if (
                existing.unit_id == unit.unit_id
                or (
                    existing.attribute == unit.attribute
                    and existing.profile_type == unit.profile_type
                    and _normalize(existing.value) == _normalize(unit.value)
                    and existing.scene == unit.scene
                )
            ):
                # Merge evidence and counts
                seen_evidence = {json.dumps(item.to_dict(), sort_keys=True, ensure_ascii=False) for item in existing.evidence}
                for item in unit.evidence:
                    marker = json.dumps(item.to_dict(), sort_keys=True, ensure_ascii=False)
                    if marker not in seen_evidence:
                        existing.evidence.append(item)
                        seen_evidence.add(marker)
                existing.session_count = max(existing.session_count, unit.session_count)
                existing.reinforcement_count = max(existing.reinforcement_count, unit.reinforcement_count)
                existing.stability_score = max(existing.stability_score, unit.stability_score)
                existing.quality_score = max(existing.quality_score, unit.quality_score)
                existing.seen_session_ids = sorted(set(existing.seen_session_ids + unit.seen_session_ids))
                existing.refresh_canonical_view()
                return
        store.append(unit)

    def _fuse_candidate(
        self,
        unit: ProfileMemoryUnit,
        candidate: ProfileCandidate,
        evidence: EvidenceItem,
        *,
        session_id: str | None,
    ) -> None:
        branch = unit.ensure_branch(
            candidate.scene,
            value=candidate.value,
            context=candidate.context,
            confidence_score=candidate.confidence,
            quality_score=candidate.quality_score,
        )
        contradiction = bool(
            candidate.value
            and branch.value
            and _similarity(candidate.value, branch.value) < 0.35
            and candidate.attribute == unit.attribute
        )
        if contradiction:
            unit.contradiction_count += 1
            branch.contradiction_count += 1

        branch.reinforcement_count += 1
        if not contradiction or candidate.confidence >= branch.confidence_score:
            branch.value = candidate.value
        branch.context = candidate.context or branch.context
        branch.confidence_score = _clamp(0.6 * branch.confidence_score + 0.4 * candidate.confidence)
        branch.quality_score = _clamp(0.55 * branch.quality_score + 0.45 * candidate.quality_score)
        branch.add_evidence(evidence)

        unit.add_evidence(evidence)
        unit.reinforcement_count += 1
        if session_id and session_id not in unit.seen_session_ids:
            unit.seen_session_ids.append(session_id)
        unit.session_count = max(1, len(unit.seen_session_ids) or unit.session_count)
        unit.confidence_score = _clamp(0.6 * unit.confidence_score + 0.4 * candidate.confidence)
        unit.quality_score = _clamp(0.55 * unit.quality_score + 0.45 * candidate.quality_score)
        observed_stability = candidate.stability + self.config.positive_reinforcement * min(
            1.0, branch.reinforcement_count / 3.0
        )
        if contradiction:
            observed_stability -= self.config.negative_penalty * 0.5
        unit.stability_score = _clamp(0.6 * unit.stability_score + 0.4 * observed_stability)
        unit.scene = candidate.scene
        unit.last_evolved = evidence.timestamp
        unit.refresh_canonical_view()

    def retrieve(self, query: str, scene: str = "general", top_k: int = 5) -> list[ProfileMemoryUnit]:
        """Retrieve from both tiers (same scoring as standard TPM)."""
        query_norm = _normalize(query)
        scored: list[tuple[float, ProfileMemoryUnit]] = []
        seen_unit_ids: set[str] = set()
        for unit in [*self.transient_memory, *self.long_term_memory]:
            if unit.unit_id in seen_unit_ids:
                continue
            branch = unit.scene_view(scene)
            rel = max(_similarity(query_norm, branch.value), _similarity(query_norm, unit.value))
            scene_score = 1.0 if branch.scene == scene else 0.7 if branch.scene == "general" or scene == "general" else 0.4
            ctx_score = max(
                _similarity(query_norm, branch.context),
                _similarity(query_norm, unit.context),
                1.0 if unit.attribute in query_norm or unit.profile_type in query_norm else 0.0,
            )
            w1, w2, w3, w4, w5 = self.config.retrieve_weights
            score = (
                w1 * rel
                + w2 * unit.stability_score
                + w3 * ctx_score
                + w4 * scene_score
                + w5 * max(unit.quality_score, branch.quality_score)
            )
            if score <= 0:
                continue
            unit.touch_access(scene)
            scored.append((score, unit))
            seen_unit_ids.add(unit.unit_id)
        scored.sort(key=lambda item: item[0], reverse=True)
        return [unit for _, unit in scored[:top_k]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": {
                "write_threshold": self.config.write_threshold,
                "context_threshold": self.config.context_threshold,
                "promote_threshold": self.config.promote_threshold,
                "promotion_min_sessions": self.config.promotion_min_sessions,
                "promote_weights": list(self.config.promote_weights),
                "retrieve_weights": list(self.config.retrieve_weights),
                "decay_lambdas": self.config.decay_lambdas,
                "positive_reinforcement": self.config.positive_reinforcement,
                "negative_penalty": self.config.negative_penalty,
                "working_decay": self.config.working_decay,
                "short_term_decay": self.config.short_term_decay,
            },
            # Map transient → short_term_memory for eval compatibility
            "working_memory": [],
            "short_term_memory": [unit.to_dict() for unit in self.transient_memory],
            "long_term_memory": [unit.to_dict() for unit in self.long_term_memory],
            "evidence_store": {
                eid: item.to_dict() for eid, item in self.evidence_store.items()
            },
            "current_session": self.current_session,
            "current_session_id": self.current_session_id,
            "architecture": "two_level",
        }


# ===== Replay engine =====

def replay_context(
    context_hash: str,
    config: TPMConfig,
    config_id: str,
    architecture: str = "standard",
) -> dict[str, Any] | None:
    """Replay all sessions for one context through TPPM with given config and architecture."""

    context_dir = CANDIDATES_DIR / context_hash
    if not context_dir.exists():
        tqdm.write(f"[WARN] No candidates for context {context_hash[:8]}")
        return None

    sessions = load_context_candidates(context_dir)
    if not sessions:
        return None

    # Select architecture
    if architecture == "flat":
        tpm = FlatTPPM(config)
    elif architecture == "two_level":
        tpm = TwoLevelTPPM(config)
    else:
        tpm = TemporalProfileMemory(config)

    skip_ltm_decay = (config_id == "ablation_decay")

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
        # Get long-term memories (architecture-dependent)
        if architecture == "flat":
            decay_units = tpm.pool  # flat pool: all units get time-shifted
        else:
            decay_units = tpm.long_term_memory

        for pmu in decay_units:
            try:
                last = datetime.fromisoformat(pmu.last_evolved)
            except (ValueError, TypeError):
                last = datetime.utcnow()
            pmu.last_evolved = (last - timedelta(hours=SESSION_INTERVAL_HOURS)).isoformat()

        if not skip_ltm_decay and architecture != "flat":
            # Flat pool has no separate long-term decay — handled in run_evolution_engine
            tpm.decay_long_term()

    snapshot = tpm.to_dict()
    snapshot["config_id"] = config_id
    snapshot["context_hash"] = context_hash
    snapshot["num_sessions"] = len(sessions)
    return snapshot


def run_replay(
    configs: list[tuple[str, TPMConfig, str]],
) -> dict[str, int]:
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

    for config_id, config, architecture in tqdm(configs, desc="Configs"):
        output_dir = SNAPSHOTS_DIR / config_id
        output_dir.mkdir(parents=True, exist_ok=True)

        n_processed = 0
        for ctx_hash in tqdm(context_hashes, desc=f"  {config_id}", leave=False):
            snapshot = replay_context(ctx_hash, config, config_id, architecture)
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
        description="Phase 2 Ablation v2: Fine-grained & hierarchy ablation replay")
    parser.add_argument("--config-id", type=str, default=None,
                        help="Single config ID (e.g. ablation_uniform_decay)")
    parser.add_argument("--dry-run", action="store_true",
                        help="List configs without running")
    args = parser.parse_args()

    configs_data = load_ablation_configs()

    # Only process new variants
    new_variant_ids = [
        "ablation_uniform_decay",
        "ablation_flat_pool",
        "ablation_two_level",
    ]

    configs: list[tuple[str, TPMConfig, str]] = []
    for cid in new_variant_ids:
        if cid not in configs_data:
            print(f"[WARN] Config {cid} not found in YAML")
            continue
        if args.config_id and cid != args.config_id:
            continue

        params = configs_data[cid]
        if not isinstance(params, dict):
            continue

        # Skip variants that reuse baseline snapshots
        if params.get("reuse_baseline"):
            print(f"[SKIP] {cid}: reuses baseline snapshots (no extraction needed)")
            continue

        config = _make_config(params)
        architecture = params.get("architecture", "standard")
        configs.append((cid, config, architecture))

    print(f"[INFO] Resolved {len(configs)} configs:")
    for cid, cfg, arch in configs:
        decay_vals = list(cfg.decay_lambdas.values())
        is_uniform = len(set(decay_vals)) <= 1
        print(f"  {cid}: arch={arch}, "
              f"decay={'uniform(' + f'{decay_vals[0]:.3f}' + ')' if is_uniform else 'type-conditioned'}, "
              f"promote_thr={cfg.promote_threshold}, "
              f"ctx_thr={cfg.context_threshold}")

    if args.dry_run:
        return 0

    stats = run_replay(configs)

    print(f"\n[DONE] Phase 2 Ablation v2 — {len(stats)} configs:")
    for cid, n in stats.items():
        print(f"  {cid}: {n} contexts")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

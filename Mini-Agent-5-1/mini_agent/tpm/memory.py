"""Temporal Profile Memory engine and persistence manager."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any
from uuid import uuid4

from .extractor import ProfileExtractor, RegexProfileExtractor
from .models import EvidenceItem, ProfileCandidate, ProfileMemoryUnit, utc_now


def _normalize(text: str) -> str:
    return " ".join((text or "").lower().split())


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def _similarity(left: str, right: str) -> float:
    left_norm = _normalize(left)
    right_norm = _normalize(right)
    if not left_norm or not right_norm:
        return 0.0
    return SequenceMatcher(None, left_norm, right_norm).ratio()


def _parse_timestamp(value: str | None) -> datetime:
    if not value:
        return utc_now()
    try:
        return datetime.fromisoformat(value)
    except Exception:
        return utc_now()


@dataclass(slots=True)
class TPMConfig:
    """Configurable TPM coefficients."""

    write_threshold: float = 0.68
    context_threshold: float = 0.62
    promote_threshold: float = 0.72
    promotion_min_sessions: int = 2
    distill_stability_threshold: float = 0.82
    distill_quality_threshold: float = 0.76
    distill_session_threshold: int = 3
    write_weights: tuple[float, float, float, float] = (0.25, 0.3, 0.25, 0.2)
    promote_weights: tuple[float, float, float, float, float] = (0.35, 0.2, 0.15, 0.25, 0.2)
    retrieve_weights: tuple[float, float, float, float, float] = (0.35, 0.2, 0.15, 0.2, 0.1)
    decay_lambdas: dict[str, float] = field(
        default_factory=lambda: {
            "goal": 0.1,
            "interest": 0.07,
            "style": 0.04,
            "background": 0.04,
            "preference": 0.05,
            "general": 0.05,
        }
    )
    positive_reinforcement: float = 0.08
    negative_penalty: float = 0.12
    working_decay: float = 0.015
    short_term_decay: float = 0.03
    # --- 论文主文对齐新增（Task 1）---
    conflict_context_threshold: float = 0.62   # δ_ctx：情境重叠阈值
    conflict_value_threshold: float = 0.35     # 极性分歧阈值
    T_fresh: float = 168.0                      # Fresh 衰减时间常数（小时）
    history_window: int = 3                     # 历史感知窗口 N


class TemporalProfileMemory:
    """Three-level TPM memory store."""

    def __init__(self, config: TPMConfig | None = None):
        self.config = config or TPMConfig()
        self.working_memory: list[ProfileMemoryUnit] = []
        self.short_term_memory: list[ProfileMemoryUnit] = []
        self.long_term_memory: list[ProfileMemoryUnit] = []
        self.evidence_store: dict[str, EvidenceItem] = {}
        self.current_session: str | None = None
        self.current_session_id: str | None = None

    def start_session(self, scene: str = "general", session_id: str | None = None) -> None:
        self.current_session = scene or "general"
        self.current_session_id = session_id or self.current_session_id
        self.working_memory = []

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
        scene = scene or self.current_session or "general"
        for unit in list(self.working_memory):
            if unit.memory_level != "working":
                continue
            unit.memory_level = "short_term"
            unit.scene = scene
            unit.refresh_canonical_view()
            self._merge_into_store(self.short_term_memory, unit)

        self.working_memory = []
        self._promote_stable_memories()
        self.current_session = scene

    def run_evolution_engine(self, scene: str = "general", *, include_long_term_decay: bool = False) -> None:
        """Run TPM evolution stages across active memory stores."""
        self._integrate_scene_views(scene)
        self._decay_store(self.working_memory, self.config.working_decay)
        self._decay_store(self.short_term_memory, self.config.short_term_decay)
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

    def retrieve(self, query: str, scene: str = "general", top_k: int = 5) -> list[ProfileMemoryUnit]:
        query_norm = _normalize(query)
        scored: list[tuple[float, ProfileMemoryUnit]] = []
        seen_unit_ids: set[str] = set()
        for unit in [*self.working_memory, *self.short_term_memory, *self.long_term_memory]:
            if unit.unit_id in seen_unit_ids:
                continue
            score = self._retrieve_score(query_norm, unit, scene)
            if score <= 0:
                continue
            unit.touch_access(scene)
            scored.append((score, unit))
            seen_unit_ids.add(unit.unit_id)
        scored.sort(key=lambda item: item[0], reverse=True)
        return [unit for _, unit in scored[:top_k]]

    def all_memories(self) -> list[ProfileMemoryUnit]:
        return [*self.short_term_memory, *self.long_term_memory]

    def distillation_candidates(self) -> list[ProfileMemoryUnit]:
        return [
            unit
            for unit in self.long_term_memory
            if unit.stability_score >= self.config.distill_stability_threshold
            and unit.quality_score >= self.config.distill_quality_threshold
            and unit.session_count >= self.config.distill_session_threshold
        ]

    def distillation_payload(self) -> list[dict[str, Any]]:
        payload: list[dict[str, Any]] = []
        for unit in self.distillation_candidates():
            branch = unit.scene_view(unit.scene)
            payload.append(
                {
                    "attribute": unit.attribute,
                    "value": branch.value,
                    "scene": branch.scene,
                    "profile_type": unit.profile_type,
                    "stability_score": unit.stability_score,
                    "quality_score": unit.quality_score,
                    "session_count": unit.session_count,
                    "evidence": [item.to_dict() for item in unit.evidence[-5:]],
                }
            )
        return payload

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": {
                "write_threshold": self.config.write_threshold,
                "context_threshold": self.config.context_threshold,
                "promote_threshold": self.config.promote_threshold,
                "promotion_min_sessions": self.config.promotion_min_sessions,
                "distill_stability_threshold": self.config.distill_stability_threshold,
                "distill_quality_threshold": self.config.distill_quality_threshold,
                "distill_session_threshold": self.config.distill_session_threshold,
                "write_weights": list(self.config.write_weights),
                "promote_weights": list(self.config.promote_weights),
                "retrieve_weights": list(self.config.retrieve_weights),
                "decay_lambdas": self.config.decay_lambdas,
                "positive_reinforcement": self.config.positive_reinforcement,
                "negative_penalty": self.config.negative_penalty,
                "working_decay": self.config.working_decay,
                "short_term_decay": self.config.short_term_decay,
                "conflict_context_threshold": self.config.conflict_context_threshold,
                "conflict_value_threshold": self.config.conflict_value_threshold,
                "T_fresh": self.config.T_fresh,
                "history_window": self.config.history_window,
            },
            "working_memory": [unit.to_dict() for unit in self.working_memory],
            "short_term_memory": [unit.to_dict() for unit in self.short_term_memory],
            "long_term_memory": [unit.to_dict() for unit in self.long_term_memory],
            "evidence_store": {
                evidence_id: item.to_dict() for evidence_id, item in self.evidence_store.items()
            },
            "current_session": self.current_session,
            "current_session_id": self.current_session_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TemporalProfileMemory":
        config_data = data.get("config", {})
        default_config = TPMConfig()
        config = TPMConfig(
            write_threshold=config_data.get("write_threshold", default_config.write_threshold),
            context_threshold=config_data.get("context_threshold", default_config.context_threshold),
            promote_threshold=config_data.get("promote_threshold", default_config.promote_threshold),
            promotion_min_sessions=config_data.get("promotion_min_sessions", default_config.promotion_min_sessions),
            distill_stability_threshold=config_data.get(
                "distill_stability_threshold", default_config.distill_stability_threshold
            ),
            distill_quality_threshold=config_data.get(
                "distill_quality_threshold", default_config.distill_quality_threshold
            ),
            distill_session_threshold=config_data.get(
                "distill_session_threshold", default_config.distill_session_threshold
            ),
            write_weights=tuple(config_data.get("write_weights", default_config.write_weights)),
            promote_weights=tuple(config_data.get("promote_weights", default_config.promote_weights)),
            retrieve_weights=tuple(config_data.get("retrieve_weights", default_config.retrieve_weights)),
            decay_lambdas=dict(config_data.get("decay_lambdas", default_config.decay_lambdas)),
            positive_reinforcement=config_data.get(
                "positive_reinforcement", default_config.positive_reinforcement
            ),
            negative_penalty=config_data.get("negative_penalty", default_config.negative_penalty),
            working_decay=config_data.get("working_decay", default_config.working_decay),
            short_term_decay=config_data.get("short_term_decay", default_config.short_term_decay),
            conflict_context_threshold=config_data.get(
                "conflict_context_threshold", default_config.conflict_context_threshold
            ),
            conflict_value_threshold=config_data.get(
                "conflict_value_threshold", default_config.conflict_value_threshold
            ),
            T_fresh=config_data.get("T_fresh", default_config.T_fresh),
            history_window=config_data.get("history_window", default_config.history_window),
        )
        memory = cls(config=config)
        memory.working_memory = [ProfileMemoryUnit.from_dict(item) for item in data.get("working_memory", [])]
        memory.short_term_memory = [ProfileMemoryUnit.from_dict(item) for item in data.get("short_term_memory", [])]
        memory.long_term_memory = [ProfileMemoryUnit.from_dict(item) for item in data.get("long_term_memory", [])]
        memory.evidence_store = {
            evidence_id: EvidenceItem.from_dict(item)
            for evidence_id, item in (data.get("evidence_store") or {}).items()
        }
        memory.current_session = data.get("current_session")
        memory.current_session_id = data.get("current_session_id")
        memory._reindex_evidence_store()
        return memory

    def save(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "TemporalProfileMemory":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)

    def evidence_for_unit(self, unit: ProfileMemoryUnit, scene: str | None = None, limit: int = 3) -> list[EvidenceItem]:
        """Return recent evidence for a PMU, optionally preferring a scene branch."""
        branch = unit.scene_view(scene or unit.scene)
        evidence = branch.evidence or unit.evidence
        return sorted(evidence, key=lambda item: item.timestamp, reverse=True)[:limit]

    def correct_memory(
        self,
        unit_id: str,
        *,
        corrected_value: str,
        correction_reason: str,
        scene: str | None = None,
        source: str = "manual_correction",
    ) -> ProfileMemoryUnit | None:
        """Revise an existing PMU while preserving correction evidence."""
        unit = self._find_unit(unit_id)
        if unit is None:
            return None
        target_scene = scene or unit.scene
        evidence = EvidenceItem(
            source=source,
            content=correction_reason,
            scene=target_scene,
            timestamp=utc_now().isoformat(),
        )
        self._register_evidence(evidence)
        branch = unit.ensure_branch(target_scene, value=corrected_value, context=correction_reason)
        branch.value = corrected_value
        branch.context = correction_reason
        branch.confidence_score = _clamp(max(branch.confidence_score, 0.9))
        branch.quality_score = _clamp(max(branch.quality_score, 0.9))
        branch.reinforcement_count += 1
        branch.add_evidence(evidence)
        unit.value = corrected_value
        unit.context = correction_reason
        unit.scene = target_scene
        unit.confidence_score = _clamp(max(unit.confidence_score, 0.9))
        unit.quality_score = _clamp(max(unit.quality_score, 0.9))
        unit.reinforcement_count += 1
        unit.add_evidence(evidence)
        unit.last_evolved = evidence.timestamp
        unit.refresh_canonical_view()
        return unit

    def _integrate_scene_views(self, scene: str) -> None:
        for unit in [*self.working_memory, *self.short_term_memory, *self.long_term_memory]:
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

    def _align_or_create(self, candidate: ProfileCandidate, session_id: str | None = None) -> ProfileMemoryUnit:
        matched = self._best_match(candidate)
        evidence = EvidenceItem(
            source=candidate.source,
            content=candidate.context,
            scene=candidate.scene,
            timestamp=candidate.timestamp,
        )
        self._register_evidence(evidence)
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
                memory_level="working",
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
            self.working_memory.append(unit)
            return unit

        self._fuse_candidate(matched, candidate, evidence, session_id=session_id)
        return matched

    def _find_unit(self, unit_id: str) -> ProfileMemoryUnit | None:
        for unit in [*self.working_memory, *self.short_term_memory, *self.long_term_memory]:
            if unit.unit_id == unit_id:
                return unit
        return None

    def _register_evidence(self, evidence: EvidenceItem) -> None:
        self.evidence_store[evidence.evidence_id] = evidence

    def _reindex_evidence_store(self) -> None:
        for unit in [*self.working_memory, *self.short_term_memory, *self.long_term_memory]:
            for item in unit.evidence:
                self._register_evidence(item)
            for branch in unit.scene_branches.values():
                for item in branch.evidence:
                    self._register_evidence(item)

    def _best_match(self, candidate: ProfileCandidate) -> ProfileMemoryUnit | None:
        stores = [*self.working_memory, *self.short_term_memory, *self.long_term_memory]
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
                self._merge_units(existing, unit)
                return
        store.append(unit)

    def _promote_stable_memories(self) -> None:
        promote_weights = self.config.promote_weights
        kept_short_term: list[ProfileMemoryUnit] = []
        for unit in self.short_term_memory:
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
                kept_short_term.append(unit)
        self.short_term_memory = kept_short_term

    def _retrieve_score(self, query_norm: str, unit: ProfileMemoryUnit, scene: str) -> float:
        branch = unit.scene_view(scene)
        rel = max(_similarity(query_norm, branch.value), _similarity(query_norm, unit.value))
        scene_score = 1.0 if branch.scene == scene else 0.7 if branch.scene == "general" or scene == "general" else 0.4
        ctx_score = max(
            _similarity(query_norm, branch.context),
            _similarity(query_norm, unit.context),
            1.0 if unit.attribute in query_norm or unit.profile_type in query_norm else 0.0,
        )
        w1, w2, w3, w4, w5 = self.config.retrieve_weights
        return (
            w1 * rel
            + w2 * unit.stability_score
            + w3 * ctx_score
            + w4 * scene_score
            + w5 * max(unit.quality_score, branch.quality_score)
        )

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

    def _merge_units(self, target: ProfileMemoryUnit, source: ProfileMemoryUnit) -> None:
        seen_evidence = {json.dumps(item.to_dict(), sort_keys=True, ensure_ascii=False) for item in target.evidence}
        for item in source.evidence:
            marker = json.dumps(item.to_dict(), sort_keys=True, ensure_ascii=False)
            if marker not in seen_evidence:
                target.evidence.append(item)
                seen_evidence.add(marker)

        for scene, branch in source.scene_branches.items():
            target_branch = target.scene_branches.get(scene)
            if target_branch is None:
                target.scene_branches[scene] = branch
                continue

            if branch.confidence_score >= target_branch.confidence_score:
                target_branch.value = branch.value
            target_branch.context = branch.context or target_branch.context
            target_branch.confidence_score = max(target_branch.confidence_score, branch.confidence_score)
            target_branch.quality_score = max(target_branch.quality_score, branch.quality_score)
            target_branch.reinforcement_count = max(target_branch.reinforcement_count, branch.reinforcement_count)
            target_branch.contradiction_count = max(target_branch.contradiction_count, branch.contradiction_count)
            target_branch.access_count = max(target_branch.access_count, branch.access_count)

            seen_branch_evidence = {
                json.dumps(item.to_dict(), sort_keys=True, ensure_ascii=False) for item in target_branch.evidence
            }
            for item in branch.evidence:
                marker = json.dumps(item.to_dict(), sort_keys=True, ensure_ascii=False)
                if marker not in seen_branch_evidence:
                    target_branch.evidence.append(item)
                    seen_branch_evidence.add(marker)

            target_branch.last_seen = max(target_branch.last_seen, branch.last_seen)
            target_branch.last_accessed = max(target_branch.last_accessed, branch.last_accessed)

        target.session_count = max(target.session_count, source.session_count)
        target.reinforcement_count = max(target.reinforcement_count, source.reinforcement_count)
        target.contradiction_count = max(target.contradiction_count, source.contradiction_count)
        target.access_count = max(target.access_count, source.access_count)
        target.stability_score = max(target.stability_score, source.stability_score)
        target.confidence_score = max(target.confidence_score, source.confidence_score)
        target.quality_score = max(target.quality_score, source.quality_score)
        target.seen_session_ids = sorted(set(target.seen_session_ids + source.seen_session_ids))
        target.last_accessed = max(target.last_accessed, source.last_accessed)
        target.last_evolved = max(target.last_evolved, source.last_evolved)
        target.refresh_canonical_view()


class TPMMemoryManager:
    """Shared TPM manager for Agent runtime and note tools."""

    def __init__(
        self,
        memory_file: str | Path,
        extractor: ProfileExtractor | None = None,
        config: TPMConfig | None = None,
        retrieval_top_k: int = 5,
    ):
        self.memory_file = Path(memory_file)
        self.extractor = extractor or RegexProfileExtractor()
        self.retrieval_top_k = retrieval_top_k
        self.config = config or TPMConfig()
        self.memory = TemporalProfileMemory(config=self.config)
        self.history_window = self.config.history_window
        self._active_scene: str | None = None
        self.session_id = f"session-{utc_now().strftime('%Y%m%dT%H%M%S')}-{uuid4().hex[:8]}"
        self._load_from_disk()

    def begin_turn(self, text: str, scene: str = "general") -> list[ProfileMemoryUnit]:
        self._load_from_disk()
        self._active_scene = scene
        self.memory.start_session(scene, session_id=self.session_id)
        candidates = self.extractor.extract(text, scene=scene)
        self.memory.ingest_candidates(candidates, scene=scene, session_id=self.session_id)
        self.memory.run_evolution_engine(scene, include_long_term_decay=False)
        self._save_to_disk()
        return self.memory.retrieve(text, scene=scene, top_k=self.retrieval_top_k)

    def augment_user_message(self, original_text: str, memories: list[ProfileMemoryUnit]) -> str:
        if not memories:
            return original_text
        scene = self._active_scene or "general"
        lines = []
        for item in memories:
            branch = item.scene_view(scene)
            evidence = self.memory.evidence_for_unit(item, scene=scene, limit=1)
            evidence_note = ""
            if evidence:
                latest = evidence[0]
                evidence_note = f", evidence_time={latest.timestamp}, evidence={latest.content[:80]}"
            lines.append(
                f"- {item.attribute}: {branch.value} "
                f"(type={item.profile_type}, scene={branch.scene}, "
                f"stability={item.stability_score:.2f}, level={item.memory_level}{evidence_note})"
            )
        return f"{original_text}\n\n[Temporal Profile Memory]\n" + "\n".join(lines)

    def complete_turn(self, scene: str | None = None) -> None:
        self._load_from_disk()
        target_scene = scene or self._active_scene or "general"
        self.memory.current_session_id = self.session_id
        self.memory.finish_session(target_scene)
        self.memory.run_evolution_engine(target_scene, include_long_term_decay=True)
        self._save_to_disk()
        self._active_scene = None

    def record_manual(self, content: str, category: str = "general") -> list[ProfileMemoryUnit]:
        self._load_from_disk()
        scene = category or "general"
        self.memory.start_session(scene, session_id=self.session_id)
        candidates = self.extractor.extract(content, scene=scene)
        if not candidates:
            candidates = [self._fallback_candidate(content, category)]
        accepted = self.memory.ingest_candidates(candidates, scene=scene, session_id=self.session_id)
        self.memory.finish_session(scene)
        self.memory.decay_long_term()
        self._save_to_disk()
        return accepted

    def recall(self, category: str | None = None, query: str | None = None, top_k: int = 10) -> list[ProfileMemoryUnit]:
        self._load_from_disk()
        if query:
            scene = category or "general"
            return self.memory.retrieve(query, scene=scene, top_k=top_k)

        memories = self.memory.all_memories()
        if category:
            category_norm = category.lower()
            filtered: list[ProfileMemoryUnit] = []
            for item in memories:
                branch_scenes = {scene_name.lower() for scene_name in item.scene_branches}
                if (
                    category_norm in item.attribute.lower()
                    or category_norm in item.profile_type.lower()
                    or category_norm == item.scene.lower()
                    or category_norm in branch_scenes
                ):
                    filtered.append(item)
            memories = filtered
        memories.sort(key=lambda item: (item.memory_level != "long_term", -item.stability_score, -item.session_count))
        return memories[:top_k]

    def format_recall(self, category: str | None = None, query: str | None = None, top_k: int = 10) -> str:
        memories = self.recall(category=category, query=query, top_k=top_k)
        if not memories:
            return "No notes recorded yet."

        scene = category or self._active_scene or "general"
        formatted = []
        for idx, item in enumerate(memories, 1):
            branch = item.scene_view(scene)
            evidence = self.memory.evidence_for_unit(item, scene=scene, limit=2)
            evidence_lines = " | ".join(f"{entry.timestamp}: {entry.content}" for entry in evidence)
            formatted.append(
                f"{idx}. [{branch.scene}] {item.attribute}: {branch.value}\n"
                f"   (type={item.profile_type}, level={item.memory_level}, "
                f"stability={item.stability_score:.2f}, sessions={item.session_count}, "
                f"branches={len(item.scene_branches)})\n"
                f"   evidence={evidence_lines}"
            )
        return "Recorded Notes:\n" + "\n".join(formatted)

    def get_memory_snapshot(self) -> dict[str, Any]:
        self._load_from_disk()
        return self.memory.to_dict()

    def export_distillation_payload(self) -> list[dict[str, Any]]:
        self._load_from_disk()
        return self.memory.distillation_payload()

    def correct_memory(
        self,
        unit_id: str,
        *,
        corrected_value: str,
        correction_reason: str,
        scene: str | None = None,
    ) -> ProfileMemoryUnit | None:
        self._load_from_disk()
        updated = self.memory.correct_memory(
            unit_id,
            corrected_value=corrected_value,
            correction_reason=correction_reason,
            scene=scene,
        )
        if updated is not None:
            self._save_to_disk()
        return updated

    def _load_from_disk(self) -> None:
        if not self.memory_file.exists():
            return
        try:
            self.memory = TemporalProfileMemory.load(self.memory_file)
        except Exception:
            self.memory = TemporalProfileMemory()

    def _save_to_disk(self) -> None:
        self.memory.save(self.memory_file)

    def _fallback_candidate(self, content: str, category: str) -> ProfileCandidate:
        category_norm = (category or "general").lower()
        profile_type_map = {
            "user_preference": "preference",
            "preference": "preference",
            "project_info": "goal",
            "decision": "goal",
            "background": "background",
            "identity": "background",
            "style": "style",
        }
        profile_type = profile_type_map.get(category_norm, "general")
        attribute = category_norm if category_norm != "general" else "explicit_note"
        return ProfileCandidate(
            attribute=attribute,
            value=content,
            context=content,
            profile_type=profile_type,
            scene=category or "general",
            confidence=0.9,
            stability=0.72,
            explicitness=0.96,
            user_relevance=0.95,
            source="manual_note",
        )

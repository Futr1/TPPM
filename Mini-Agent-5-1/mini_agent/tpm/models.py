"""Temporal Profile Memory data models."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any
from uuid import uuid4


def utc_now() -> datetime:
    """Return a stable UTC timestamp."""
    return datetime.utcnow()


# 论文表1：7 个心理子空间 a_i
SLOT_VALUES = {"affect", "stressor", "cognitive", "coping", "support", "behavior", "risk"}
# 论文式(15)：5 个时间–安全类型 g_i
MEMORY_TYPE_VALUES = {"affect", "stressor", "coping", "support", "trait"}

# §4.1 默认 slot -> memory_type 映射（抽取器只给 slot 时回填 g_i）
DEFAULT_MEMORY_TYPE = {
    "affect": "affect",
    "stressor": "stressor",
    "coping": "coping",
    "support": "support",
    "cognitive": "trait",
    "behavior": "coping",
    "risk": "affect",
}

# §4.2 旧 profile_type -> {slot, memory_type} 迁移表（仅 from_dict 内存解析，不回写）
LEGACY_PROFILE_TYPE_MAP = {
    "background": ("support", "trait"),
    "preference": ("cognitive", "trait"),
    "goal": ("behavior", "coping"),
    "style": ("cognitive", "trait"),
    "interest": ("behavior", "trait"),
    "general": ("coping", "trait"),
}


def default_memory_type(slot: str) -> str:
    """按 §4.1 回填 memory_type；未知 slot 默认 trait。"""
    return DEFAULT_MEMORY_TYPE.get(slot, "trait")


def migrate_profile_type(legacy: str) -> tuple[str, str]:
    """把旧 profile_type（或裸 slot/g_i）解析为 (slot, memory_type)。best-effort。"""
    legacy = (legacy or "general").strip().lower()
    if legacy in LEGACY_PROFILE_TYPE_MAP:
        return LEGACY_PROFILE_TYPE_MAP[legacy]
    if legacy in SLOT_VALUES:
        return (legacy, default_memory_type(legacy))
    if legacy in MEMORY_TYPE_VALUES:
        return ("coping", legacy)
    return ("coping", "trait")


@dataclass(slots=True)
class EvidenceItem:
    """Evidence supporting a profile memory unit."""

    source: str
    content: str
    scene: str
    evidence_id: str = field(default_factory=lambda: uuid4().hex)
    timestamp: str = field(default_factory=lambda: utc_now().isoformat())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvidenceItem":
        return cls(**data)


@dataclass(slots=True)
class ProfileCandidate:
    """Candidate profile item extracted from conversation."""

    attribute: str
    value: str
    context: str
    slot: str
    memory_type: str
    scene: str = "general"
    confidence: float = 0.7
    stability: float = 0.5
    relevance: float = 1.0
    explicitness: float = 0.7
    utility: float = 0.75
    source: str = "user_utterance"
    timestamp: str = field(default_factory=lambda: utc_now().isoformat())

    def write_score(self, weights: tuple[float, float, float, float]) -> float:
        """Compute the write score (论文式8: φ = α1·r + α2·e + α3·u + α4·b)."""
        alpha1, alpha2, alpha3, alpha4 = weights
        return (
            alpha1 * self.relevance
            + alpha2 * self.explicitness
            + alpha3 * self.utility
            + alpha4 * self.stability
        )

    @property
    def quality_score(self) -> float:
        """Quality proxy used by explicit TPM rules."""
        return (self.confidence + self.explicitness + self.utility) / 3.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProfileCandidate":
        payload = dict(data)
        if "slot" not in payload:
            legacy = payload.pop("profile_type", "general")
            payload["slot"], payload["memory_type"] = migrate_profile_type(legacy)
        elif "memory_type" not in payload:
            payload["memory_type"] = default_memory_type(payload["slot"])
        payload.pop("profile_type", None)
        # 重命名因子向后兼容
        if "relevance" not in payload and "recency" in payload:
            payload["relevance"] = payload.pop("recency")
        if "utility" not in payload and "user_relevance" in payload:
            payload["utility"] = payload.pop("user_relevance")
        payload.pop("recency", None)
        payload.pop("user_relevance", None)
        return cls(**payload)


@dataclass(slots=True)
class SceneProfileBranch:
    """Scene-conditioned branch stored inside a PMU."""

    scene: str
    value: str
    context: str
    confidence_score: float
    quality_score: float
    evidence: list[EvidenceItem] = field(default_factory=list)
    reinforcement_count: int = 1
    contradiction_count: int = 0
    access_count: int = 0
    last_seen: str = field(default_factory=lambda: utc_now().isoformat())
    last_accessed: str = field(default_factory=lambda: utc_now().isoformat())

    def add_evidence(self, item: EvidenceItem) -> None:
        self.evidence.append(item)
        self.last_seen = item.timestamp
        self.last_accessed = item.timestamp

    def touch_access(self) -> None:
        self.access_count += 1
        self.last_accessed = utc_now().isoformat()

    def to_dict(self) -> dict[str, Any]:
        return {
            "scene": self.scene,
            "value": self.value,
            "context": self.context,
            "confidence_score": self.confidence_score,
            "quality_score": self.quality_score,
            "evidence": [item.to_dict() for item in self.evidence],
            "reinforcement_count": self.reinforcement_count,
            "contradiction_count": self.contradiction_count,
            "access_count": self.access_count,
            "last_seen": self.last_seen,
            "last_accessed": self.last_accessed,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SceneProfileBranch":
        payload = dict(data)
        payload["evidence"] = [EvidenceItem.from_dict(item) for item in payload.get("evidence", [])]
        return cls(**payload)


@dataclass(slots=True)
class ProfileMemoryUnit:
    """Profile Memory Unit (PMU) used by TPM."""

    attribute: str
    value: str
    context: str
    slot: str
    memory_type: str
    stability_score: float
    confidence_score: float
    scene: str
    quality_score: float
    evidence: list[EvidenceItem] = field(default_factory=list)
    scene_branches: dict[str, SceneProfileBranch] = field(default_factory=dict)
    unit_id: str = field(default_factory=lambda: uuid4().hex)
    timestamp: str = field(default_factory=lambda: utc_now().isoformat())
    session_count: int = 1
    reinforcement_count: int = 1
    contradiction_count: int = 0
    access_count: int = 0
    seen_session_ids: list[str] = field(default_factory=list)
    last_accessed: str = field(default_factory=lambda: utc_now().isoformat())
    last_evolved: str = field(default_factory=lambda: utc_now().isoformat())
    memory_level: str = "working"

    @property
    def is_risk(self) -> bool:
        """论文风险安全规则：slot 属于风险信号子空间时为真。"""
        return self.slot == "risk"

    def ensure_branch(
        self,
        scene: str,
        *,
        value: str | None = None,
        context: str | None = None,
        confidence_score: float | None = None,
        quality_score: float | None = None,
    ) -> SceneProfileBranch:
        """Get or create the scene-conditioned branch for a PMU."""
        scene_key = (scene or "general").strip() or "general"
        branch = self.scene_branches.get(scene_key)
        if branch is None:
            branch = SceneProfileBranch(
                scene=scene_key,
                value=value or self.value,
                context=context or self.context,
                confidence_score=self.confidence_score if confidence_score is None else confidence_score,
                quality_score=self.quality_score if quality_score is None else quality_score,
            )
            self.scene_branches[scene_key] = branch
        return branch

    def add_evidence(self, item: EvidenceItem) -> None:
        self.evidence.append(item)
        self.last_accessed = item.timestamp
        self.last_evolved = item.timestamp

    def touch_access(self, scene: str = "general") -> None:
        self.access_count += 1
        self.last_accessed = utc_now().isoformat()
        branch = self.scene_branches.get(scene) or self.scene_branches.get("general")
        if branch is None and self.scene_branches:
            branch = self.scene_view(scene)
        if branch is not None:
            branch.touch_access()

    def scene_view(self, scene: str = "general") -> SceneProfileBranch:
        """Return the best scene-conditioned branch for retrieval/rendering."""
        if scene in self.scene_branches:
            return self.scene_branches[scene]
        if "general" in self.scene_branches:
            return self.scene_branches["general"]
        if self.scene_branches:
            return max(
                self.scene_branches.values(),
                key=lambda branch: (
                    branch.reinforcement_count,
                    branch.quality_score,
                    branch.access_count,
                ),
            )

        branch = SceneProfileBranch(
            scene=self.scene or "general",
            value=self.value,
            context=self.context,
            confidence_score=self.confidence_score,
            quality_score=self.quality_score,
        )
        self.scene_branches[branch.scene] = branch
        return branch

    def refresh_canonical_view(self) -> None:
        """Refresh the PMU's dominant scene/value/context from its branches."""
        dominant = self.scene_view(self.scene or "general")
        for branch in self.scene_branches.values():
            candidate_score = (
                branch.reinforcement_count
                + branch.quality_score
                + 0.25 * branch.access_count
                - 0.2 * branch.contradiction_count
            )
            dominant_score = (
                dominant.reinforcement_count
                + dominant.quality_score
                + 0.25 * dominant.access_count
                - 0.2 * dominant.contradiction_count
            )
            if candidate_score > dominant_score:
                dominant = branch

        self.scene = dominant.scene
        self.value = dominant.value
        self.context = dominant.context
        self.confidence_score = max(self.confidence_score, dominant.confidence_score)
        self.quality_score = max(self.quality_score, dominant.quality_score)

    def to_dict(self) -> dict[str, Any]:
        return {
            "attribute": self.attribute,
            "value": self.value,
            "context": self.context,
            "slot": self.slot,
            "memory_type": self.memory_type,
            "stability_score": self.stability_score,
            "confidence_score": self.confidence_score,
            "scene": self.scene,
            "quality_score": self.quality_score,
            "evidence": [item.to_dict() for item in self.evidence],
            "scene_branches": {
                scene: branch.to_dict() for scene, branch in self.scene_branches.items()
            },
            "unit_id": self.unit_id,
            "timestamp": self.timestamp,
            "session_count": self.session_count,
            "reinforcement_count": self.reinforcement_count,
            "contradiction_count": self.contradiction_count,
            "access_count": self.access_count,
            "seen_session_ids": self.seen_session_ids,
            "last_accessed": self.last_accessed,
            "last_evolved": self.last_evolved,
            "memory_level": self.memory_level,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProfileMemoryUnit":
        payload = dict(data)
        payload["evidence"] = [EvidenceItem.from_dict(item) for item in payload.get("evidence", [])]

        if "slot" not in payload:
            legacy = payload.pop("profile_type", "general")
            payload["slot"], payload["memory_type"] = migrate_profile_type(legacy)
        elif "memory_type" not in payload:
            payload["memory_type"] = default_memory_type(payload["slot"])
        payload.pop("profile_type", None)

        raw_branches = payload.get("scene_branches") or {}
        payload["scene_branches"] = {
            scene: SceneProfileBranch.from_dict(branch) for scene, branch in raw_branches.items()
        }

        unit = cls(
            attribute=payload["attribute"],
            value=payload["value"],
            context=payload.get("context", ""),
            slot=payload["slot"],
            memory_type=payload["memory_type"],
            stability_score=payload.get("stability_score", 0.5),
            confidence_score=payload.get("confidence_score", 0.7),
            scene=payload.get("scene", "general"),
            quality_score=payload.get("quality_score", 0.6),
            evidence=payload.get("evidence", []),
            scene_branches=payload.get("scene_branches", {}),
            unit_id=payload.get("unit_id", uuid4().hex),
            timestamp=payload.get("timestamp", utc_now().isoformat()),
            session_count=payload.get("session_count", 1),
            reinforcement_count=payload.get("reinforcement_count", 1),
            contradiction_count=payload.get("contradiction_count", 0),
            access_count=payload.get("access_count", 0),
            seen_session_ids=list(payload.get("seen_session_ids", [])),
            last_accessed=payload.get("last_accessed", utc_now().isoformat()),
            last_evolved=payload.get("last_evolved", payload.get("last_accessed", utc_now().isoformat())),
            memory_level=payload.get("memory_level", "working"),
        )

        if not unit.scene_branches:
            legacy_scene = payload.get("scene", "general")
            unit.scene_branches[legacy_scene] = SceneProfileBranch(
                scene=legacy_scene,
                value=unit.value,
                context=unit.context,
                confidence_score=unit.confidence_score,
                quality_score=unit.quality_score,
                evidence=list(unit.evidence),
                reinforcement_count=unit.reinforcement_count,
                contradiction_count=unit.contradiction_count,
                access_count=unit.access_count,
                last_seen=unit.last_evolved,
                last_accessed=unit.last_accessed,
            )

        if not unit.seen_session_ids and unit.session_count > 0:
            unit.seen_session_ids = [f"legacy-session-{index + 1}" for index in range(unit.session_count)]

        unit.refresh_canonical_view()
        return unit

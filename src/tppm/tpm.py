"""Backward-compatibility shim for mini_agent.tpm → tppm.core migration.

Provides the mini_agent.tpm namespace so existing benchmark scripts
continue to work without modification.
"""

from tppm.core.extractor import LLMProfileExtractor, ProfileExtractor, RegexProfileExtractor
from tppm.core.memory import TPMConfig, TPMMemoryManager, TemporalProfileMemory
from tppm.core.models import EvidenceItem, ProfileCandidate, ProfileMemoryUnit, SceneProfileBranch

__all__ = [
    "EvidenceItem",
    "LLMProfileExtractor",
    "ProfileCandidate",
    "ProfileExtractor",
    "ProfileMemoryUnit",
    "RegexProfileExtractor",
    "SceneProfileBranch",
    "TPMConfig",
    "TPMMemoryManager",
    "TemporalProfileMemory",
]

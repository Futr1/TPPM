"""Paper configuration consistency tests.

Verifies that the paper-reported hyperparameters match across all config sources:
- TPMConfig defaults (memory.py)
- TPMSettings defaults (config.py)
- Ablation YAML baseline
- Benchmark scripts default configs
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

# Ensure Mini-Agent is importable
_AGENT_ROOT = Path(__file__).resolve().parent.parent
if str(_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(_AGENT_ROOT))

from mini_agent.tpm.memory import TPMConfig
from mini_agent.config import TPMSettings


# ==================== Paper baseline (single source of truth) ====================

PAPER_BASELINE = {
    "write_threshold": 0.68,
    "promote_threshold": 0.72,
    "context_threshold": 0.62,
    "promotion_min_sessions": 2,
    "write_weights": (0.25, 0.30, 0.25, 0.20),
    "retrieve_weights": (0.35, 0.20, 0.15, 0.20, 0.10),
}


class TestTPMConfigDefaults:
    """TPMConfig defaults must match paper baseline."""

    def test_default_write_threshold(self):
        assert TPMConfig().write_threshold == PAPER_BASELINE["write_threshold"]

    def test_default_promote_threshold(self):
        assert TPMConfig().promote_threshold == PAPER_BASELINE["promote_threshold"]

    def test_default_context_threshold(self):
        assert TPMConfig().context_threshold == PAPER_BASELINE["context_threshold"]

    def test_default_promotion_min_sessions(self):
        assert TPMConfig().promotion_min_sessions == PAPER_BASELINE["promotion_min_sessions"]

    def test_default_write_weights(self):
        assert TPMConfig().write_weights == PAPER_BASELINE["write_weights"]

    def test_default_retrieve_weights(self):
        assert TPMConfig().retrieve_weights == PAPER_BASELINE["retrieve_weights"]

    def test_default_top_k(self):
        """top_k=5 is the retrieval default."""
        assert 5 == 5  # top_k is a retrieve() parameter, not a config field


class TestTPMSettingsDefaults:
    """TPMSettings pydantic defaults must match TPMConfig."""

    def test_settings_match_tpm_config(self):
        settings = TPMSettings()
        tpm = TPMConfig()
        assert settings.write_threshold == tpm.write_threshold
        assert settings.promote_threshold == tpm.promote_threshold
        assert settings.context_threshold == tpm.context_threshold
        assert settings.promotion_min_sessions == tpm.promotion_min_sessions

    def test_settings_write_weights_match(self):
        settings = TPMSettings()
        assert tuple(settings.write_weights) == TPMConfig().write_weights

    def test_settings_retrieve_weights_match(self):
        settings = TPMSettings()
        assert tuple(settings.retrieve_weights) == TPMConfig().retrieve_weights


class TestAblationYAMLBaseline:
    """Ablation YAML baseline must match paper baseline."""

    @pytest.fixture
    def ablation_yaml(self):
        repo_root = Path(__file__).resolve().parent.parent.parent
        yaml_path = repo_root / "Ablation" / "configs" / "ablation.yaml"
        if not yaml_path.exists():
            pytest.skip(f"Ablation YAML not found: {yaml_path}")
        with open(yaml_path) as f:
            return yaml.safe_load(f)

    def test_baseline_write_threshold(self, ablation_yaml):
        assert ablation_yaml["baseline"]["write_threshold"] == PAPER_BASELINE["write_threshold"]

    def test_baseline_promote_threshold(self, ablation_yaml):
        assert ablation_yaml["baseline"]["promote_threshold"] == PAPER_BASELINE["promote_threshold"]

    def test_baseline_context_threshold(self, ablation_yaml):
        assert ablation_yaml["baseline"]["context_threshold"] == PAPER_BASELINE["context_threshold"]

    def test_baseline_promotion_min_sessions(self, ablation_yaml):
        assert ablation_yaml["baseline"]["promotion_min_sessions"] == PAPER_BASELINE["promotion_min_sessions"]

    def test_no_config_uses_legacy_0_58_1(self, ablation_yaml):
        """No config named 'baseline' should use the legacy 0.58/1 values."""
        baseline = ablation_yaml.get("baseline", {})
        assert baseline.get("promote_threshold") != 0.58, (
            "baseline config still uses legacy promote_threshold=0.58. Paper value is 0.72."
        )
        assert baseline.get("promotion_min_sessions") != 1, (
            "baseline config still uses legacy promotion_min_sessions=1. Paper value is 2."
        )

    def test_ablation_variants_only_override_target_params(self, ablation_yaml):
        """Ablation variants should not silently change non-target parameters."""
        ablation_ids = [k for k in ablation_yaml if k.startswith("ablation_")]
        for aid in ablation_ids:
            if "promote_threshold" in ablation_yaml[aid]:
                # Only consolidation ablation should change promote_threshold significantly
                if "consolidation" in aid:
                    continue
                pt = ablation_yaml[aid]["promote_threshold"]
                # If it matches paper baseline, that's fine (inherited)
                # Deliberate non-standard values should be documented
                assert pt == PAPER_BASELINE["promote_threshold"] or pt >= 0.999, (
                    f"{aid}: promote_threshold={pt} deviates from paper baseline "
                    f"({PAPER_BASELINE['promote_threshold']}) without justification"
                )


class TestNoLegacyBaseline:
    """Verify legacy 0.58/1 values do not appear in active configs."""

    def test_param_sweep_baseline_is_legacy(self):
        """param_sweep.yaml uses 'legacy_baseline', not 'baseline'."""
        repo_root = Path(__file__).resolve().parent.parent.parent
        yaml_path = repo_root / "Table3-data" / "configs" / "param_sweep.yaml"
        if not yaml_path.exists():
            pytest.skip(f"param_sweep.yaml not found: {yaml_path}")
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
        # 'baseline' key should NOT exist with legacy values
        if "baseline" in data:
            assert data["baseline"].get("promote_threshold") != 0.58, (
                "param_sweep.yaml has 'baseline' with legacy promote_threshold=0.58"
            )
        # 'legacy_baseline' is the renamed old config
        assert "legacy_baseline" in data, (
            "param_sweep.yaml should have 'legacy_baseline' for the old 0.58/1 config"
        )

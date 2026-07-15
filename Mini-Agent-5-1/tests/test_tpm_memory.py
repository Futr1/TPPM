"""Tests for TPM-backed Mini-Agent memory."""

from datetime import timedelta
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import pytest

from mini_agent.agent import Agent
from mini_agent.schema import LLMResponse
from mini_agent.tpm import TPMMemoryManager, TemporalProfileMemory
from mini_agent.tpm.extractor import LLMProfileExtractor, RegexProfileExtractor
from mini_agent.tpm.models import ProfileCandidate, utc_now
from mini_agent.tools.note_tool import RecallNoteTool, SessionNoteTool


class DummyLLM:
    async def generate(self, messages, tools=None):
        return LLMResponse(content=str(messages[-1].content), finish_reason="stop")


class DummyHTTPResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def make_test_workspace() -> Path:
    workspace = Path(__file__).resolve().parents[1] / ".tmp-smoke" / f"tpm-test-{uuid4().hex}"
    workspace.mkdir(parents=True, exist_ok=True)
    return workspace


def test_tpm_manager_persists_and_recalls():
    workspace = make_test_workspace()
    memory_file = workspace / ".agent_memory.json"
    manager = TPMMemoryManager(memory_file=memory_file)

    manager.record_manual("User prefers concise responses", category="user_preference")
    manager.record_manual("Project uses Python 3.12", category="project_info")

    assert memory_file.exists()
    recalled = manager.format_recall()
    assert "concise responses" in recalled
    assert "Python 3.12" in recalled


def test_llm_profile_extractor_parses_structured_candidates():
    extractor = LLMProfileExtractor(
        api_key="dummy",
        api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model="qwen2.5-7b-instruct",
        fallback_extractor=RegexProfileExtractor(),
    )

    payload = {
        "choices": [
            {
                "message": {
                    "content": """
                    {
                      "candidates": [
                        {
                          "attribute": "style",
                          "value": "concise answers",
                          "context": "User explicitly prefers concise answers",
                          "slot": "cognitive",
                          "memory_type": "trait",
                          "scene": "coding",
                          "confidence": 0.92,
                          "stability": 0.8,
                          "relevance": 1.0,
                          "explicitness": 0.95,
                          "utility": 0.94,
                          "source": "llm_qwen"
                        }
                      ]
                    }
                    """
                }
            }
        ]
    }

    with patch("mini_agent.tpm.extractor.requests.post", return_value=DummyHTTPResponse(payload)):
        candidates = extractor.extract("I prefer concise answers.", scene="coding")

    assert len(candidates) == 1
    assert candidates[0].attribute == "style"
    assert candidates[0].value == "concise answers"
    assert candidates[0].slot == "cognitive"
    assert candidates[0].memory_type == "trait"
    assert candidates[0].scene == "coding"


def test_tpm_scene_branches_and_session_count_are_explicit():
    memory = TemporalProfileMemory()

    session_a = "session-a"
    memory.start_session("coding", session_id=session_a)
    memory.ingest_candidates(
        [
            ProfileCandidate(
                attribute="style",
                value="concise answers",
                context="User prefers concise answers while coding",
                slot="cognitive",
                memory_type="trait",
                scene="coding",
                confidence=0.92,
                stability=0.84,
                explicitness=0.95,
                utility=0.93,
            )
        ],
        scene="coding",
        session_id=session_a,
    )
    memory.finish_session("coding")

    memory.start_session("research", session_id=session_a)
    memory.ingest_candidates(
        [
            ProfileCandidate(
                attribute="style",
                value="concise answers",
                context="User still prefers concise answers in research planning",
                slot="cognitive",
                memory_type="trait",
                scene="research",
                confidence=0.9,
                stability=0.82,
                explicitness=0.94,
                utility=0.9,
            )
        ],
        scene="research",
        session_id=session_a,
    )
    memory.finish_session("research")

    assert len(memory.short_term_memory) == 1
    unit = memory.short_term_memory[0]
    assert unit.session_count == 1
    assert "coding" in unit.scene_branches
    assert "research" in unit.scene_branches

    session_b = "session-b"
    memory.start_session("coding", session_id=session_b)
    memory.ingest_candidates(
        [
            ProfileCandidate(
                attribute="style",
                value="concise answers",
                context="User repeats concise answers preference",
                slot="cognitive",
                memory_type="trait",
                scene="coding",
                confidence=0.91,
                stability=0.85,
                explicitness=0.95,
                utility=0.92,
            )
        ],
        scene="coding",
        session_id=session_b,
    )
    memory.finish_session("coding")

    merged = memory.all_memories()[0]
    assert merged.session_count == 2


def test_tpm_manager_counts_distinct_sessions_not_turns():
    workspace = make_test_workspace()
    memory_file = workspace / ".agent_memory.json"

    manager = TPMMemoryManager(memory_file=memory_file)
    manager.record_manual("User prefers concise responses", category="style")
    manager.record_manual("User prefers concise responses", category="style")

    snapshot = manager.get_memory_snapshot()
    memories = snapshot.get("short_term_memory", []) + snapshot.get("long_term_memory", [])
    assert memories
    assert memories[0]["session_count"] == 1

    manager2 = TPMMemoryManager(memory_file=memory_file)
    manager2.record_manual("User prefers concise responses", category="style")

    snapshot2 = manager2.get_memory_snapshot()
    memories2 = snapshot2.get("short_term_memory", []) + snapshot2.get("long_term_memory", [])
    assert memories2[0]["session_count"] == 2


def test_tpm_evidence_store_keeps_traceable_supporting_utterances():
    workspace = make_test_workspace()
    memory_file = workspace / ".agent_memory.json"
    manager = TPMMemoryManager(memory_file=memory_file)

    manager.record_manual("User prefers concise responses", category="style")
    snapshot = manager.get_memory_snapshot()
    memories = snapshot.get("short_term_memory", []) + snapshot.get("long_term_memory", [])
    evidence_store = snapshot.get("evidence_store", {})

    assert memories
    assert evidence_store
    unit_evidence = memories[0]["evidence"]
    assert unit_evidence
    evidence_id = unit_evidence[0]["evidence_id"]
    assert evidence_id in evidence_store
    assert evidence_store[evidence_id]["content"] == "User prefers concise responses"
    assert evidence_store[evidence_id]["timestamp"]


def test_tpm_retrieve_includes_working_memory_before_session_finish():
    workspace = make_test_workspace()
    memory_file = workspace / ".agent_memory.json"
    manager = TPMMemoryManager(memory_file=memory_file)

    memories = manager.begin_turn("I prefer concise answers for this project.", scene="coding")

    assert memories
    assert any(item.attribute == "style" for item in memories)
    snapshot = manager.get_memory_snapshot()
    assert snapshot.get("working_memory")


def test_tpm_memory_correction_preserves_auditable_evidence():
    workspace = make_test_workspace()
    memory_file = workspace / ".agent_memory.json"
    manager = TPMMemoryManager(memory_file=memory_file)

    manager.record_manual("User prefers concise responses", category="style")
    snapshot = manager.get_memory_snapshot()
    memories = snapshot.get("short_term_memory", []) + snapshot.get("long_term_memory", [])
    unit_id = memories[0]["unit_id"]
    initial_evidence_count = len(snapshot.get("evidence_store", {}))

    updated = manager.correct_memory(
        unit_id,
        corrected_value="detailed responses",
        correction_reason="User later clarified they want more detailed responses.",
        scene="style",
    )

    assert updated is not None
    snapshot2 = manager.get_memory_snapshot()
    memories2 = snapshot2.get("short_term_memory", []) + snapshot2.get("long_term_memory", [])
    assert memories2[0]["value"] == "detailed responses"
    assert len(snapshot2.get("evidence_store", {})) == initial_evidence_count + 1
    assert any(
        item["content"] == "User later clarified they want more detailed responses."
        for item in snapshot2["evidence_store"].values()
    )


def test_llm_profile_extractor_falls_back_to_regex_when_request_fails():
    extractor = LLMProfileExtractor(
        api_key="dummy",
        api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model="qwen2.5-7b-instruct",
        fallback_extractor=RegexProfileExtractor(),
    )

    with patch("mini_agent.tpm.extractor.requests.post", side_effect=RuntimeError("network error")):
        candidates = extractor.extract("I like multi-agent systems.", scene="research")

    assert candidates
    assert any(item.attribute == "interest" for item in candidates)


@pytest.mark.asyncio
async def test_tpm_note_tools_keep_legacy_tool_names():
    workspace = make_test_workspace()
    memory_file = workspace / ".agent_memory.json"
    manager = TPMMemoryManager(memory_file=memory_file)

    record_tool = SessionNoteTool(memory_manager=manager)
    recall_tool = RecallNoteTool(memory_manager=manager)

    record_result = await record_tool.execute("I like multi-agent systems", category="research")
    recall_result = await recall_tool.execute(category="research")

    assert record_result.success
    assert recall_result.success
    assert "multi-agent systems" in recall_result.content


@pytest.mark.asyncio
async def test_agent_injects_temporal_profile_memory_context():
    workspace = make_test_workspace()
    memory_file = workspace / ".agent_memory.json"
    manager = TPMMemoryManager(memory_file=memory_file)
    manager.record_manual("User prefers concise answers", category="style")

    agent = Agent(
        llm_client=DummyLLM(),
        system_prompt="You are a helpful assistant.",
        tools=[],
        workspace_dir=workspace,
        memory_manager=manager,
    )

    agent.add_user_message("Please help me write code.", scene="style")
    result = await agent.run()

    assert "[Temporal Profile Memory]" in result
    assert "concise answers" in result


def test_tpm_config_externalized_from_yaml(tmp_path):
    from mini_agent.config import Config, build_tpm_config

    yaml_text = """
api_key: "local"
provider: "openai"
model: "deepseek-v4-flash"
max_steps: 10
workspace_dir: ./ws
tpm:
  write_threshold: 0.7
  context_threshold: 0.6
  conflict_context_threshold: 0.55
  conflict_value_threshold: 0.3
  T_fresh: 96.0
  history_window: 5
tools:
  enable_note: true
"""
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml_text, encoding="utf-8")

    config = Config.from_yaml(cfg_path)
    assert config.tpm.T_fresh == 96.0
    assert config.tpm.history_window == 5
    assert config.tpm.conflict_context_threshold == 0.55

    tpm_config = build_tpm_config(config.tpm)
    assert tpm_config.T_fresh == 96.0
    assert tpm_config.history_window == 5
    assert tpm_config.conflict_value_threshold == 0.3
    # weights round-trip list -> tuple
    assert isinstance(tpm_config.write_weights, tuple)
    assert len(tpm_config.write_weights) == 4
    # decay_lambdas omitted in yaml -> falls back to TPMConfig defaults
    from mini_agent.tpm.memory import TPMConfig
    assert tpm_config.decay_lambdas == TPMConfig().decay_lambdas


def test_tpm_manager_exposes_history_window_from_config():
    from mini_agent.tpm import TPMMemoryManager
    from mini_agent.tpm.memory import TPMConfig

    workspace = make_test_workspace()
    manager = TPMMemoryManager(
        memory_file=workspace / ".agent_memory.json",
        config=TPMConfig(history_window=7, T_fresh=120.0),
    )
    assert manager.history_window == 7
    assert manager.config.T_fresh == 120.0


def test_tpm_config_roundtrips_new_fields_through_to_dict():
    from mini_agent.tpm import TemporalProfileMemory
    from mini_agent.tpm.memory import TPMConfig

    memory = TemporalProfileMemory(config=TPMConfig(T_fresh=72.0, history_window=4))
    data = memory.to_dict()
    assert data["config"]["T_fresh"] == 72.0
    assert data["config"]["history_window"] == 4
    assert data["config"]["conflict_context_threshold"] == TPMConfig().conflict_context_threshold

    restored = TemporalProfileMemory.from_dict(data)
    assert restored.config.T_fresh == 72.0
    assert restored.config.history_window == 4


def test_profile_candidate_from_legacy_profile_type_migrates_to_dual_fields():
    from mini_agent.tpm.models import ProfileCandidate

    candidate = ProfileCandidate.from_dict({
        "attribute": "style",
        "value": "concise answers",
        "context": "User prefers concise answers",
        "profile_type": "style",
        "scene": "coding",
        "confidence": 0.9,
        "stability": 0.8,
        "recency": 1.0,
        "explicitness": 0.9,
        "user_relevance": 0.9,
    })
    assert candidate.slot == "cognitive"
    assert candidate.memory_type == "trait"
    assert candidate.relevance == 1.0
    assert candidate.utility == 0.9
    assert not hasattr(candidate, "profile_type")


def test_profile_memory_unit_from_legacy_profile_type_migrates_and_is_risk():
    from mini_agent.tpm.models import ProfileMemoryUnit

    unit = ProfileMemoryUnit.from_dict({
        "attribute": "mood",
        "value": "anxious",
        "context": "user reports anxiety",
        "profile_type": "general",
        "stability_score": 0.6,
        "confidence_score": 0.7,
        "scene": "general",
        "quality_score": 0.6,
    })
    assert unit.slot == "coping"
    assert unit.memory_type == "trait"
    assert unit.is_risk is False

    risk_unit = ProfileMemoryUnit.from_dict({
        "attribute": "self_harm",
        "value": "ideation present",
        "context": "user mentions self-harm",
        "slot": "risk",
        "memory_type": "affect",
        "stability_score": 0.6,
        "confidence_score": 0.9,
        "scene": "general",
        "quality_score": 0.7,
    })
    assert risk_unit.is_risk is True


def test_profile_unit_to_dict_writes_dual_fields_not_profile_type():
    from mini_agent.tpm.models import ProfileMemoryUnit

    unit = ProfileMemoryUnit(
        attribute="mood", value="calm", context="ok",
        slot="affect", memory_type="affect",
        stability_score=0.6, confidence_score=0.7, scene="general", quality_score=0.6,
    )
    data = unit.to_dict()
    assert "slot" in data and data["slot"] == "affect"
    assert "memory_type" in data and data["memory_type"] == "affect"
    assert "profile_type" not in data


def test_default_memory_type_backfill_for_each_slot():
    from mini_agent.tpm.models import default_memory_type

    assert default_memory_type("affect") == "affect"
    assert default_memory_type("stressor") == "stressor"
    assert default_memory_type("cognitive") == "trait"
    assert default_memory_type("behavior") == "coping"
    assert default_memory_type("risk") == "affect"


def test_risk_unit_skips_time_decay_but_drops_on_contradiction():
    from mini_agent.tpm import TemporalProfileMemory
    from mini_agent.tpm.memory import TPMConfig
    from mini_agent.tpm.models import ProfileMemoryUnit

    memory = TemporalProfileMemory(config=TPMConfig())
    old_time = (utc_now() - timedelta(days=30)).isoformat()

    risk_unit = ProfileMemoryUnit(
        attribute="self_harm", value="ideation", context="risk signal",
        slot="risk", memory_type="affect",
        stability_score=0.8, confidence_score=0.9, scene="general", quality_score=0.7,
        last_evolved=old_time, last_accessed=old_time,
        memory_level="long_term", session_count=2,
    )
    memory.long_term_memory.append(risk_unit)

    memory.decay_long_term()
    # 风险单元跳过常规时间衰减，contradiction=0 -> 强度不变
    assert risk_unit.stability_score == pytest.approx(0.8, abs=1e-6)

    # 命中 contradiction 后降强度
    risk_unit.contradiction_count = 2
    risk_unit.reinforcement_count = 1
    risk_unit.last_evolved = old_time
    memory.decay_long_term()
    expected = max(0.0, min(1.0, 0.8 - memory.config.negative_penalty * 1.0))
    assert risk_unit.stability_score == pytest.approx(expected, abs=1e-6)
    assert risk_unit.stability_score < 0.8


def test_non_risk_affect_unit_decays_over_time():
    from mini_agent.tpm import TemporalProfileMemory
    from mini_agent.tpm.memory import TPMConfig
    from mini_agent.tpm.models import ProfileMemoryUnit

    memory = TemporalProfileMemory(config=TPMConfig())
    old_time = (utc_now() - timedelta(days=30)).isoformat()
    unit = ProfileMemoryUnit(
        attribute="mood", value="anxious", context="anxiety",
        slot="affect", memory_type="affect",
        stability_score=0.8, confidence_score=0.7, scene="general", quality_score=0.6,
        last_evolved=old_time, last_accessed=old_time,
        memory_level="long_term", session_count=2, reinforcement_count=2,
    )
    memory.long_term_memory.append(unit)
    memory.decay_long_term()
    # affect 衰减率最大，30 天后强度应明显下降
    assert unit.stability_score < 0.8


def test_decay_lambdas_default_keys_are_g_i_per_paper_eq15():
    from mini_agent.tpm.memory import TPMConfig

    cfg = TPMConfig()
    # spec §5.1：decay_lambdas 键改为 g_i（不再是旧 profile_type 键）
    assert set(cfg.decay_lambdas.keys()) == {"affect", "stressor", "coping", "support", "trait"}
    # 论文式(15) 排序：affect > stressor > coping ≈ support > trait
    assert cfg.decay_lambdas["affect"] > cfg.decay_lambdas["stressor"]
    assert cfg.decay_lambdas["stressor"] > cfg.decay_lambdas["coping"]
    assert cfg.decay_lambdas["coping"] == cfg.decay_lambdas["support"]
    assert cfg.decay_lambdas["support"] > cfg.decay_lambdas["trait"]

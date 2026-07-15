"""Tests for TPM-backed Mini-Agent memory."""

from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import pytest

from mini_agent.agent import Agent
from mini_agent.schema import LLMResponse
from mini_agent.tpm import TPMMemoryManager, TemporalProfileMemory
from mini_agent.tpm.extractor import LLMProfileExtractor, RegexProfileExtractor
from mini_agent.tpm.models import ProfileCandidate
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
                          "profile_type": "style",
                          "scene": "coding",
                          "confidence": 0.92,
                          "stability": 0.8,
                          "recency": 1.0,
                          "explicitness": 0.95,
                          "user_relevance": 0.94,
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
    assert candidates[0].profile_type == "style"
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
                profile_type="style",
                scene="coding",
                confidence=0.92,
                stability=0.84,
                explicitness=0.95,
                user_relevance=0.93,
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
                profile_type="style",
                scene="research",
                confidence=0.9,
                stability=0.82,
                explicitness=0.94,
                user_relevance=0.9,
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
                profile_type="style",
                scene="coding",
                confidence=0.91,
                stability=0.85,
                explicitness=0.95,
                user_relevance=0.92,
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

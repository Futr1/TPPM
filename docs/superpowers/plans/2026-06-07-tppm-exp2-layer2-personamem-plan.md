# Experiment 2 Layer 2 PersonaMem Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement three-phase TPPM parameter sensitivity analysis pipeline on PersonaMem 32K benchmark.

**Architecture:** Phase 1 (DeepSeek LLM) extracts ProfileCandidates from 37 shared contexts (4–7 sessions each, detected via `role=system` boundaries). Phase 2 (pure Python) replays sessions through TemporalProfileMemory for each parameter config. Phase 3 (vLLM) evaluates QA accuracy per config by injecting formatted TPPM memory into the 32K context window.

**Tech Stack:** Python 3.10+, asyncio, DeepSeek API (deepseek-v4-flash), vLLM (OpenAI-compatible HTTP), Qwen3.5-9B, Mini-Agent-5-1 TPPM engine

---

## File Structure

```
Table3-data/
├── configs/
│   └── param_sweep.yaml              # Parameter sweep definitions (Task 1)
├── scripts/
│   ├── phase1_extract_candidates.py  # Session detection + LLM extraction (Task 2)
│   ├── phase2_replay_evolution.py    # Memory replay with parameter sweeps (Task 3)
│   ├── phase3_eval_qa.py             # vLLM-based QA evaluation (Task 4)
│   └── summarize.py                  # Aggregate results across configs (Task 5)
├── candidates/                        # Phase 1 output (auto-created)
├── memory_snapshots/                  # Phase 2 output (auto-created)
├── eval_results/                      # Phase 3 output (auto-created)
```

---

### Task 1: Parameter Sweep Configuration

**Files:**
- Create: `/root/autodl-tmp/wangqihao/Table3-data/configs/param_sweep.yaml`

- [ ] **Step 1: Write `configs/param_sweep.yaml`**

```yaml
# TPPM Experiment 2 Layer 2 — Parameter Sweep Definitions
# Each sweep defines a list of configs with a unique config_id.

# Baseline config (default TPMConfig values)
baseline:
  write_threshold: 0.68
  promote_threshold: 0.72
  context_threshold: 0.62
  decay_lambdas:
    goal: 0.1
    interest: 0.07
    style: 0.04
    background: 0.04
    preference: 0.05
    general: 0.05

# Sub-experiment 2a: Consolidation threshold scan
# Vary write_threshold while holding other params at baseline
sweep_2a_write:
  description: "Consolidation write_threshold sensitivity"
  mechanism: consolidation
  variable_param: write_threshold
  configs:
    - config_id: write_0.56
      write_threshold: 0.56
    - config_id: write_0.62
      write_threshold: 0.62
    - config_id: write_0.68
      write_threshold: 0.68
    - config_id: write_0.74
      write_threshold: 0.74
    - config_id: write_0.80
      write_threshold: 0.80

sweep_2a_promote:
  description: "Consolidation promote_threshold sensitivity"
  mechanism: consolidation
  variable_param: promote_threshold
  configs:
    - config_id: promote_0.60
      promote_threshold: 0.60
    - config_id: promote_0.66
      promote_threshold: 0.66
    - config_id: promote_0.72
      promote_threshold: 0.72
    - config_id: promote_0.78
      promote_threshold: 0.78
    - config_id: promote_0.84
      promote_threshold: 0.84

# Sub-experiment 2b: Decay lambda scan
# Global scale factor applied to all 6 per-type decay_lambdas
sweep_2b_decay:
  description: "Decay lambda sensitivity (global scale)"
  mechanism: decay
  variable_param: decay_lambdas_scale
  configs:
    - config_id: decay_0.25x
      decay_lambdas_scale: 0.25
    - config_id: decay_0.5x
      decay_lambdas_scale: 0.5
    - config_id: decay_1.0x
      decay_lambdas_scale: 1.0
    - config_id: decay_2.0x
      decay_lambdas_scale: 2.0
    - config_id: decay_4.0x
      decay_lambdas_scale: 4.0

# Sub-experiment 2c: Branching context_threshold scan
sweep_2c_context:
  description: "Branching context_threshold sensitivity"
  mechanism: branching
  variable_param: context_threshold
  configs:
    - config_id: ctx_0.50
      context_threshold: 0.50
    - config_id: ctx_0.56
      context_threshold: 0.56
    - config_id: ctx_0.62
      context_threshold: 0.62
    - config_id: ctx_0.68
      context_threshold: 0.68
    - config_id: ctx_0.74
      context_threshold: 0.74
```

- [ ] **Step 2: Commit**

```bash
cd /root/autodl-tmp/wangqihao
git add Table3-data/configs/param_sweep.yaml
git commit -m "feat: add param_sweep.yaml for Exp2 Layer 2 sensitivity analysis"
```

---

### Task 2: Phase 1 — Candidate Extraction Script

**Files:**
- Create: `/root/autodl-tmp/wangqihao/Table3-data/scripts/phase1_extract_candidates.py`

**Dependencies:** `locomo_tppm_extract.py` (DeepSeek API patterns, candidate parsing), `Mini-Agent-5-1/mini_agent/tpm/models.py` (ProfileCandidate)

- [ ] **Step 1: Write `phase1_extract_candidates.py`**

```python
#!/usr/bin/env python3
"""Phase 1: Extract ProfileCandidates from PersonaMem shared contexts via DeepSeek API.

Session boundary detection: role=system messages split the flat message list
into chronological sessions.

Usage:
    python3 phase1_extract_candidates.py                           # all contexts
    python3 phase1_extract_candidates.py --max-contexts 2          # smoke test
    python3 phase1_extract_candidates.py --context-id <hash>       # single context
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Any

# Allow importing Mini-Agent-5-1 TPPM modules
_AGENT_ROOT = Path("/root/autodl-tmp/wangqihao/Mini-Agent-5-1")
if str(_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(_AGENT_ROOT))

from mini_agent.tpm.models import ProfileCandidate

from openai import AsyncOpenAI
from tqdm import tqdm

# ===== Paths =====
ROOT = Path("/root/autodl-tmp/wangqihao/Table3-data")
DATASETS = Path("/root/autodl-tmp/wangqihao/datasets/PersonaMem")
SHARED_CONTEXTS_PATH = DATASETS / "shared_contexts_32k.jsonl"
CANDIDATES_DIR = ROOT / "candidates"

# ===== DeepSeek API Config =====
API_BASE = "https://api.deepseek.com"
API_MODEL = "deepseek-v4-flash"
API_KEY = "REDACTED_DEEPSEEK_KEY"

CONCURRENCY = 8
MAX_RETRIES = 5
REQUEST_TIMEOUT = 60.0
MAX_TOKENS = 2048
INITIAL_BACKOFF = 1.0
MAX_BACKOFF = 30.0


# ===== Data loading =====

def load_shared_contexts(path: Path) -> list[tuple[str, list[dict]]]:
    """Load all shared contexts from JSONL.

    Returns:
        list of (context_hash, messages_list) tuples.
    """
    contexts: list[tuple[str, list[dict]]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            # Each line is {hash: [messages]}
            for key, msgs in obj.items():
                contexts.append((key, msgs))
    return contexts


def detect_sessions(messages: list[dict]) -> list[tuple[int, list[dict]]]:
    """Split flat message list into sessions at role=system boundaries.

    Each system message marks the start of a new session. The system message
    itself is included as the first message of the session.

    Returns:
        list of (session_idx, session_messages) sorted chronologically.
    """
    sessions: list[tuple[int, list[dict]]] = []
    current_session: list[dict] = []

    for msg in messages:
        if msg.get("role") == "system" and current_session:
            # System message starts a new session — save previous
            sessions.append((len(sessions), current_session))
            current_session = [msg]
        else:
            current_session.append(msg)

    # Don't forget the last session
    if current_session:
        sessions.append((len(sessions), current_session))

    return sessions


def format_session_for_extraction(session_messages: list[dict]) -> str:
    """Format session messages into a single dialogue text block for LLM extraction."""
    lines: list[str] = []
    for msg in session_messages:
        role = str(msg.get("role", "")).strip()
        content = str(msg.get("content", "")).strip()
        if not content:
            continue
        if role == "system":
            # Skip persona description prefix — keep it brief
            if "persona:" in content.lower() and len(content) > 300:
                continue
            lines.append(f"[Context] {content}")
        elif role == "user":
            lines.append(f"User: {content}")
        elif role == "assistant":
            lines.append(f"Assistant: {content}")
    return "\n".join(lines)


# ===== Async LLM extraction =====

def build_extraction_payload(dialogue_text: str, scene: str = "general") -> dict[str, Any]:
    """Build DeepSeek API payload for profile candidate extraction."""
    schema_hint = {
        "candidates": [
            {
                "attribute": "short_attribute_name",
                "value": "profile_value",
                "context": "supporting_span_or_short_reason",
                "profile_type": "background|preference|goal|style|interest|general",
                "scene": scene,
                "confidence": 0.0,
                "stability": 0.0,
                "recency": 1.0,
                "explicitness": 0.0,
                "user_relevance": 0.0,
                "source": "llm_deepseek",
            }
        ]
    }
    system_prompt = (
        "You are a profile candidate extractor for Temporal Profile Memory (TPM). "
        "Extract stable, reusable, and scene-conditioned user profile information "
        "from the latest conversation session. "
        "Return ONLY valid JSON, no markdown, no explanation."
    )
    user_prompt = (
        "Task: extract profile candidates for TPM.\n"
        f"Current scene: {scene}\n"
        f"Latest conversation session:\n{dialogue_text}\n\n"
        "Extraction rules:\n"
        "1. Keep only user-related profile facts, preferences, goals, style tendencies, "
        "identity, or stable context. Focus on information about the speakers (not the assistant).\n"
        "2. Ignore generic conversational filler and greetings.\n"
        "3. Use concise attribute names like identity, interest, preference, "
        "current_goal, style, project_focus, personal_background.\n"
        "4. profile_type must be one of: background, preference, goal, style, interest, general.\n"
        "5. confidence, stability, recency, explicitness, user_relevance must be numbers in [0,1].\n"
        "6. user_relevance measures how central this fact is to the user's enduring profile.\n"
        "7. Prefer higher stability for repeated or enduring traits; "
        "lower stability for short-term goals.\n"
        "8. If there is no useful profile memory candidate, return {\"candidates\": []}.\n\n"
        f"Output JSON schema example:\n{json.dumps(schema_hint, ensure_ascii=False)}"
    )
    return {
        "model": API_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0,
        "max_tokens": MAX_TOKENS,
        "response_format": {"type": "json_object"},
    }


def parse_candidates_from_response(
    content: str, scene: str, original_text: str
) -> list[dict[str, Any]]:
    """Parse LLM JSON response into candidate dicts.

    Returns list of dicts with ProfileCandidate-compatible fields.
    """
    stripped = content.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?", "", stripped, flags=re.IGNORECASE).strip()
        stripped = re.sub(r"```$", "", stripped).strip()

    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        first = stripped.find("{")
        last = stripped.rfind("}")
        if first != -1 and last != -1 and last > first:
            try:
                parsed = json.loads(stripped[first:last + 1])
            except json.JSONDecodeError:
                return []
        else:
            return []

    if isinstance(parsed, dict):
        raw_list = parsed.get("candidates", [])
    elif isinstance(parsed, list):
        raw_list = parsed
    else:
        return []

    def _clamp(value: Any, default: float) -> float:
        try:
            return max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            return default

    def _default_stability(ptype: str) -> float:
        defaults = {
            "background": 0.9, "style": 0.78, "preference": 0.72,
            "interest": 0.7, "goal": 0.56, "general": 0.6,
        }
        return defaults.get(ptype, 0.6)

    candidates: list[dict[str, Any]] = []
    for item in raw_list:
        if not isinstance(item, dict):
            continue
        attr = str(item.get("attribute", "")).strip()
        val = str(item.get("value", "")).strip()
        if not attr or not val:
            continue
        ptype = str(item.get("profile_type", "general")).strip().lower()
        if ptype not in {"background", "preference", "goal", "style", "interest", "general"}:
            ptype = "general"
        candidates.append({
            "attribute": attr,
            "value": val,
            "context": str(item.get("context") or original_text).strip() or original_text,
            "profile_type": ptype,
            "scene": str(item.get("scene") or scene).strip() or scene,
            "confidence": _clamp(item.get("confidence"), 0.72),
            "stability": _clamp(item.get("stability"), _default_stability(ptype)),
            "recency": _clamp(item.get("recency"), 1.0),
            "explicitness": _clamp(item.get("explicitness"), 0.8),
            "user_relevance": _clamp(item.get("user_relevance"), 0.82),
            "source": str(item.get("source") or "llm_deepseek").strip() or "llm_deepseek",
        })
    return candidates


async def extract_candidates_async(
    client: AsyncOpenAI,
    dialogue_text: str,
    scene: str = "general",
    context_hash: str = "",
    session_idx: int = 0,
) -> list[dict[str, Any]]:
    """Async call to DeepSeek for profile extraction with retries."""
    if not dialogue_text.strip():
        return []

    payload = build_extraction_payload(dialogue_text, scene)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = await client.chat.completions.create(
                model=API_MODEL,
                temperature=0,
                max_tokens=MAX_TOKENS,
                response_format={"type": "json_object"},
                messages=payload["messages"],
            )
            content = resp.choices[0].message.content or ""
            if not content.strip():
                continue
            candidates = parse_candidates_from_response(content, scene, dialogue_text)
            return candidates[:8]
        except Exception:
            if attempt >= MAX_RETRIES:
                raise
            sleep_s = min(MAX_BACKOFF, INITIAL_BACKOFF * (2 ** (attempt - 1)))
            sleep_s += random.uniform(0.0, 0.25 * sleep_s)
            await asyncio.sleep(sleep_s)

    return []


# ===== Per-context processing =====

async def process_context(
    context_hash: str,
    messages: list[dict],
    client: AsyncOpenAI,
    sem: asyncio.Semaphore,
) -> tuple[str, int, int]:
    """Extract candidates for all sessions in one shared context.

    Returns:
        (context_hash, num_sessions, num_candidates_extracted)
    """
    sessions = detect_sessions(messages)
    if not sessions:
        return context_hash, 0, 0

    output_dir = CANDIDATES_DIR / context_hash
    output_dir.mkdir(parents=True, exist_ok=True)

    total_candidates = 0

    for session_idx, session_msgs in sessions:
        dialogue_text = format_session_for_extraction(session_msgs)
        scene = f"session_{session_idx}"

        try:
            async with sem:
                candidates = await extract_candidates_async(
                    client, dialogue_text, scene=scene,
                    context_hash=context_hash, session_idx=session_idx,
                )
        except Exception as exc:
            tqdm.write(f"[ERROR] {context_hash[:8]} session {session_idx}: {exc}")
            candidates = []

        output_path = output_dir / f"session_{session_idx:03d}.json"
        with output_path.open("w", encoding="utf-8") as f:
            json.dump({
                "context_hash": context_hash,
                "session_idx": session_idx,
                "session_id": f"{context_hash}_session_{session_idx}",
                "scene": scene,
                "dialogue_text": dialogue_text,
                "candidates": candidates,
            }, f, ensure_ascii=False, indent=2)

        total_candidates += len(candidates)

    return context_hash, len(sessions), total_candidates


# ===== Main runner =====

async def run_extraction(
    contexts: list[tuple[str, list[dict]]],
    concurrency: int = CONCURRENCY,
) -> tuple[int, int, int]:
    """Run candidate extraction across all contexts concurrently."""
    client = AsyncOpenAI(api_key=API_KEY, base_url=API_BASE, timeout=REQUEST_TIMEOUT)
    sem = asyncio.Semaphore(concurrency)

    tasks = [
        process_context(ctx_hash, msgs, client, sem)
        for ctx_hash, msgs in contexts
    ]

    total_contexts = 0
    total_sessions = 0
    total_candidates = 0

    progress = tqdm(asyncio.as_completed(tasks), total=len(tasks),
                    desc="Extracting TPPM candidates")
    for coro in progress:
        ctx_hash, num_sessions, num_candidates = await coro
        total_contexts += 1
        total_sessions += num_sessions
        total_candidates += num_candidates
        progress.set_postfix({
            "ctx": ctx_hash[:8],
            "sessions": num_sessions,
            "cands": num_candidates,
        })

    return total_contexts, total_sessions, total_candidates


# ===== CLI =====

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Phase 1: Extract TPPM candidates from PersonaMem shared contexts")
    parser.add_argument("--input", type=Path, default=SHARED_CONTEXTS_PATH)
    parser.add_argument("--max-contexts", type=int, default=None,
                        help="Limit number of contexts (for smoke testing)")
    parser.add_argument("--context-id", type=str, default=None,
                        help="Process a single context by hash prefix")
    parser.add_argument("--concurrency", type=int, default=CONCURRENCY)
    args = parser.parse_args()

    contexts = load_shared_contexts(args.input)
    if args.context_id:
        contexts = [(h, m) for h, m in contexts if h.startswith(args.context_id)]
    if args.max_contexts:
        contexts = contexts[:args.max_contexts]

    total_sessions = sum(len(detect_sessions(msgs)) for _, msgs in contexts)
    print(f"[INFO] Contexts: {len(contexts)}")
    print(f"[INFO] Total sessions: {total_sessions}")
    print(f"[INFO] Model: {API_MODEL}")
    print(f"[INFO] Concurrency: {args.concurrency}")

    n_ctx, n_sessions, n_cands = asyncio.run(
        run_extraction(contexts, concurrency=args.concurrency)
    )

    print(f"\n[DONE] Processed {n_ctx} contexts, {n_sessions} sessions, {n_cands} candidates")
    print(f"[DONE] Output: {CANDIDATES_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Verify script parses correctly**

```bash
cd /root/autodl-tmp/wangqihao/Table3-data
python3 -c "import ast; ast.parse(open('scripts/phase1_extract_candidates.py').read()); print('Syntax OK')"
```

- [ ] **Step 3: Smoke test with 1 context**

```bash
cd /root/autodl-tmp/wangqihao/Table3-data
python3 scripts/phase1_extract_candidates.py --max-contexts 1
```

Expected: Creates `candidates/<hash>/session_000.json` through `session_00N.json` with candidate lists.

- [ ] **Step 4: Commit**

```bash
cd /root/autodl-tmp/wangqihao
git add Table3-data/scripts/phase1_extract_candidates.py
git commit -m "feat: add Phase 1 candidate extraction for PersonaMem

Detects sessions via role=system boundaries, extracts ProfileCandidates
via DeepSeek API with async concurrency. Based on locomo_tppm_extract.py."
```

---

### Task 3: Phase 2 — Memory Evolution Replay Script

**Files:**
- Create: `/root/autodl-tmp/wangqihao/Table3-data/scripts/phase2_replay_evolution.py`

**Dependencies:** `Mini-Agent-5-1/mini_agent/tpm/memory.py` (TemporalProfileMemory, TPMConfig), `Mini-Agent-5-1/mini_agent/tpm/models.py` (ProfileCandidate), `configs/param_sweep.yaml`

- [ ] **Step 1: Write `phase2_replay_evolution.py`**

```python
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
from pathlib import Path
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
ROOT = Path("/root/autodl-tmp/wangqihao/Table3-data")
CANDIDATES_DIR = ROOT / "candidates"
SNAPSHOTS_DIR = ROOT / "memory_snapshots"
SWEEP_CONFIG_PATH = ROOT / "configs" / "param_sweep.yaml"


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
        promote_threshold=float(params.get("promote_threshold", 0.72)),
        context_threshold=float(params.get("context_threshold", 0.62)),
        decay_lambdas=decay_lambdas,
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
            tpm.ingest_candidates(candidates, scene=scene, session_id=session_id)

        tpm.finish_session(scene=scene)

    # Run long-term decay after all sessions
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
        print(f"  {cid}: write_thr={cfg.write_threshold}, promote_thr={cfg.promote_threshold}, "
              f"ctx_thr={cfg.context_threshold}, decay_scale≈{cfg.decay_lambdas.get('goal',0.1)/0.1:.2f}x")

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
```

- [ ] **Step 2: Verify script parses correctly**

```bash
cd /root/autodl-tmp/wangqihao/Table3-data
python3 -c "import ast; ast.parse(open('scripts/phase2_replay_evolution.py').read()); print('Syntax OK')"
```

- [ ] **Step 3: Dry-run to verify config resolution**

```bash
cd /root/autodl-tmp/wangqihao/Table3-data
python3 scripts/phase2_replay_evolution.py --dry-run
```

Expected: Lists all configs with resolved parameter values (baseline + 4 sweeps × 5 levels = 21 configs).

- [ ] **Step 4: Test with a single config on existing candidates**

```bash
cd /root/autodl-tmp/wangqihao/Table3-data
python3 scripts/phase2_replay_evolution.py --config-id baseline
```

Expected: Creates `memory_snapshots/baseline/<hash>.json` for each context that has Phase 1 candidates.

- [ ] **Step 5: Commit**

```bash
cd /root/autodl-tmp/wangqihao
git add Table3-data/scripts/phase2_replay_evolution.py
git commit -m "feat: add Phase 2 memory evolution replay with parameter sweeps

Replays all Phase 1 candidates through TPPM for each config.
Supports --config-id, --sweep, and --dry-run flags. Pure Python."
```

---

### Task 4: Phase 3 — QA Evaluation Script

**Files:**
- Create: `/root/autodl-tmp/wangqihao/Table3-data/scripts/phase3_eval_qa.py`

**Dependencies:** vLLM (OpenAI-compatible HTTP server), PersonaMem CSV + JSONL data, Phase 2 memory snapshots

- [ ] **Step 1: Write `phase3_eval_qa.py`**

```python
#!/usr/bin/env python3
"""Phase 3: Evaluate QA accuracy for each TPPM config using vLLM + Qwen3.5-9B.

Builds a 32K-token context window: conversation history + TPPM memory + question.
Queries vLLM via OpenAI-compatible HTTP API.

Usage:
    python3 phase3_eval_qa.py --config-id baseline
    python3 phase3_eval_qa.py --config-id baseline --max-questions 10   # smoke test
    python3 phase3_eval_qa.py --config-id baseline --no-tppm             # ablation

vLLM server must be running:
    vllm serve Qwen/Qwen3.5-9B --tensor-parallel-size 2 --max-model-len 32768
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import tiktoken
from openai import OpenAI
from tqdm import tqdm

# ===== Paths =====
ROOT = Path("/root/autodl-tmp/wangqihao/Table3-data")
DATASETS = Path("/root/autodl-tmp/wangqihao/datasets/PersonaMem")
QUESTIONS_CSV = DATASETS / "questions_32k.csv"
SHARED_CONTEXTS_JSONL = DATASETS / "shared_contexts_32k.jsonl"
SNAPSHOTS_DIR = ROOT / "memory_snapshots"
EVAL_DIR = ROOT / "eval_results"

# ===== vLLM Config =====
VLLM_BASE_URL = "http://localhost:8000/v1"
VLLM_MODEL = "Qwen/Qwen3.5-9B"
MAX_CONTEXT_TOKENS = 32768
MEMORY_TOKEN_BUDGET = 2048  # Max tokens for TPPM memory block

# ===== Tokenizer =====
TOKENIZER = tiktoken.encoding_for_model("gpt-4o")  # Approximate token counter


# ===== JSONL Index =====

def build_jsonl_index(jsonl_path: Path) -> dict[str, int]:
    """Build file-offset index for JSONL: {key: byte_offset}."""
    index: dict[str, int] = {}
    with jsonl_path.open("r", encoding="utf-8") as f:
        while True:
            offset = f.tell()
            line = f.readline()
            if not line:
                break
            key = next(iter(json.loads(line).keys()))
            index[key] = offset
    return index


def load_context_by_id(jsonl_path: Path, offset: int) -> list[dict]:
    """Load a single shared context from JSONL by byte offset."""
    with jsonl_path.open("r", encoding="utf-8") as f:
        f.seek(offset)
        item = json.loads(f.readline())
        return next(iter(item.values()))


# ===== Context window builder =====

def format_memory_block(memory_snapshot: dict[str, Any], max_tokens: int) -> str:
    """Format TPPM long-term memory into a compact text block.

    Sorts PMUs by stability_score * confidence_score descending,
    then takes top entries that fit within max_tokens budget.

    Args:
        memory_snapshot: Phase 2 output dict with long_term_memory list.
        max_tokens: Token budget for the memory block.

    Returns:
        Formatted memory string.
    """
    long_term = memory_snapshot.get("long_term_memory", [])
    if not long_term:
        return ""

    # Score and sort PMUs
    scored: list[tuple[float, dict]] = []
    for pmu in long_term:
        stability = float(pmu.get("stability_score", 0))
        quality = float(pmu.get("quality_score", 0))
        score = stability * quality
        scored.append((score, pmu))
    scored.sort(key=lambda x: x[0], reverse=True)

    # Build entries, tracking token budget
    header = "[TPPM Memory — structured user profile]\n"
    header_tokens = len(TOKENIZER.encode(header))
    budget = max_tokens - header_tokens

    entries: list[str] = []
    for _, pmu in scored:
        attribute = pmu.get("attribute", "?")
        value = pmu.get("canonical_value", pmu.get("value", "?"))
        profile_type = pmu.get("profile_type", "general")
        stability = float(pmu.get("stability_score", 0))
        quality = float(pmu.get("quality_score", 0))
        scene = pmu.get("scene", "general")

        entry = (
            f"- {attribute}: {value} "
            f"(type={profile_type}, stability={stability:.2f}, quality={quality:.2f}, scene={scene})"
        )
        entry_tokens = len(TOKENIZER.encode(entry))

        if budget - entry_tokens < 0:
            break

        entries.append(entry)
        budget -= entry_tokens

    if not entries:
        return ""

    return header + "\n".join(entries) + "\n"


def build_context_window(
    conversation: list[dict],
    end_index: int,
    memory_snapshot: dict[str, Any] | None,
    question: str,
    all_options: str,
) -> list[dict]:
    """Build the 32K-token context window for vLLM.

    Order: conversation (truncated) → memory block → question + options.

    Args:
        conversation: Full shared context message list.
        end_index: Cutoff index from PersonaMem question (exclusive).
        memory_snapshot: Phase 2 memory snapshot, or None for no-TPPM ablation.
        question: The QA question text.
        all_options: Formatted options string "(a) ... (b) ... (c) ... (d) ..."

    Returns:
        List of {"role": ..., "content": ...} dicts for vLLM chat API.
    """
    instructions = (
        "Find the most appropriate model response and give your final answer "
        "(a), (b), (c), or (d) after the special token <final_answer>."
    )

    # Truncate conversation to end_index
    conv = conversation[:end_index]

    # Count tokens for the fixed parts
    question_block = f"{question}\n\n{instructions}\n\n{all_options}"
    question_tokens = len(TOKENIZER.encode(question_block))

    # Build conversation as text, count tokens
    conv_text = _messages_to_text(conv)
    conv_tokens = len(TOKENIZER.encode(conv_text))

    # Calculate memory budget
    available = MAX_CONTEXT_TOKENS - question_tokens - conv_tokens

    memory_block = ""
    if memory_snapshot is not None and available > 200:
        memory_budget = min(MEMORY_TOKEN_BUDGET, available - 100)
        memory_block = format_memory_block(memory_snapshot, memory_budget)

    # Re-count after memory is added, and truncate conversation if needed
    memory_tokens = len(TOKENIZER.encode(memory_block)) if memory_block else 0
    total_used = conv_tokens + memory_tokens + question_tokens

    if total_used > MAX_CONTEXT_TOKENS:
        # Truncate conversation to fit
        excess = total_used - MAX_CONTEXT_TOKENS + 200  # extra margin
        # Crude truncation: drop roughly excess tokens from conversation
        conv_text_chars = len(conv_text)
        trunc_ratio = max(0, (conv_text_chars - excess * 4) / max(1, conv_text_chars))
        conv_text = conv_text[:int(len(conv_text) * trunc_ratio)]
        conv_text += "\n[... conversation truncated to fit context window ...]"

    # Assemble final messages
    system_content = "You are a helpful assistant answering questions about a user based on conversation history and profile memory."

    user_content_parts = []
    if conv_text:
        user_content_parts.append(f"[Conversation History]\n{conv_text}")
    if memory_block:
        user_content_parts.append(memory_block)
    user_content_parts.append(question_block)

    user_content = "\n\n".join(user_content_parts)

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]


def _messages_to_text(messages: list[dict]) -> str:
    """Convert message list to compact text format."""
    lines: list[str] = []
    for msg in messages:
        role = str(msg.get("role", "")).strip()
        content = str(msg.get("content", "")).strip()
        if not content:
            continue
        if role == "system":
            # Truncate persona descriptions — keep only first 2 sentences
            if len(content) > 500:
                sentences = re.split(r'(?<=[.!?])\s+', content)
                content = " ".join(sentences[:2])
                if len(sentences) > 2:
                    content += " [...]"
            lines.append(f"[Context] {content}")
        elif role == "user":
            lines.append(f"User: {content}")
        elif role == "assistant":
            lines.append(f"Assistant: {content}")
    return "\n".join(lines)


# ===== Answer extraction =====

def extract_answer(predicted_answer: str, correct_answer: str) -> tuple[bool, str]:
    """Extract predicted option and compare against correct answer.

    Ported from PersonaMem inference_standalone_openai.py.
    """
    def _extract_only_options(text: str) -> set[str]:
        text = text.lower()
        in_parens = re.findall(r'\(([a-d])\)', text)
        if in_parens:
            return set(in_parens)
        else:
            return set(re.findall(r'\b([a-d])\b', text))

    correct = correct_answer.lower().strip("() ")

    full_response = predicted_answer
    predicted_answer = predicted_answer.strip()
    if "<final_answer>" in predicted_answer:
        predicted_answer = predicted_answer.split("<final_answer>")[-1].strip()
    if predicted_answer.endswith("</final_answer>"):
        predicted_answer = predicted_answer[:-len("</final_answer>")].strip()

    pred_options = _extract_only_options(predicted_answer)
    if pred_options == {correct}:
        return True, predicted_answer

    # Fallback: search full response
    response_options = _extract_only_options(full_response)
    if response_options == {correct}:
        return True, predicted_answer

    return False, predicted_answer


# ===== Evaluation runner =====

def run_evaluation(
    config_id: str,
    no_tppm: bool = False,
    max_questions: int | None = None,
    vllm_url: str = VLLM_BASE_URL,
) -> tuple[Path, int, int]:
    """Run QA evaluation for a single config.

    Args:
        config_id: Config ID from Phase 2 (e.g. 'baseline', 'write_0.56').
        no_tppm: If True, skip TPPM memory (ablation baseline).
        max_questions: Limit questions for smoke testing.
        vllm_url: vLLM server URL.

    Returns:
        (output_path, num_correct, num_total)
    """
    client = OpenAI(base_url=vllm_url, api_key="not-needed")

    # Build JSONL index for shared contexts
    jsonl_index = build_jsonl_index(SHARED_CONTEXTS_JSONL)

    # Load memory snapshots for this config (unless no_tppm)
    memory_cache: dict[str, dict] = {}
    if not no_tppm:
        snapshot_dir = SNAPSHOTS_DIR / config_id
        if snapshot_dir.exists():
            for fpath in snapshot_dir.glob("*.json"):
                with fpath.open("r", encoding="utf-8") as f:
                    snapshot = json.load(f)
                ctx_hash = snapshot.get("context_hash", fpath.stem)
                memory_cache[ctx_hash] = snapshot
        else:
            print(f"[WARN] No snapshots found for config '{config_id}' at {snapshot_dir}")

    # Output path
    output_dir = EVAL_DIR / config_id
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "results.csv"

    # Read questions
    total_correct = 0
    total_questions = 0
    prev_sid = None
    prev_context = None

    with open(output_path, "w", newline="", encoding="utf-8") as out_f:
        writer = csv.writer(out_f)
        writer.writerow([
            "score", "persona_id", "question_id", "question_type", "topic",
            "correct_answer", "predicted_answer", "model_response",
            "config_id", "context_length_in_tokens",
        ])

        with open(QUESTIONS_CSV, "r", newline="", encoding="utf-8") as csv_f:
            reader = csv.DictReader(csv_f)
            for row in tqdm(reader, desc=f"Evaluating {config_id}", total=max_questions or 589):
                if max_questions and total_questions >= max_questions:
                    break

                total_questions += 1

                sid = row["shared_context_id"]
                end_index = int(row["end_index_in_shared_context"])

                # Load shared context (cache by sid)
                if sid != prev_sid:
                    if sid in jsonl_index:
                        prev_context = load_context_by_id(SHARED_CONTEXTS_JSONL, jsonl_index[sid])
                    else:
                        prev_context = []
                    prev_sid = sid
                context = prev_context

                # Get memory snapshot for this context
                memory = memory_cache.get(sid) if not no_tppm else None

                # Build context window
                question_text = row["user_question_or_message"]
                all_options = row["all_options"]
                correct_answer = row["correct_answer"]

                messages = build_context_window(
                    context, end_index, memory,
                    question_text, all_options,
                )

                try:
                    response = client.chat.completions.create(
                        model=VLLM_MODEL,
                        messages=messages,
                        max_tokens=256,
                        temperature=0,
                    )
                    model_response = response.choices[0].message.content or ""
                except Exception as e:
                    tqdm.write(f"[ERROR] LLM call failed: {e}")
                    model_response = ""

                score, predicted = extract_answer(model_response, correct_answer)
                if score:
                    total_correct += 1

                writer.writerow([
                    score,
                    row["persona_id"],
                    row["question_id"],
                    row["question_type"],
                    row["topic"],
                    correct_answer,
                    predicted,
                    model_response[:500],  # Truncate long responses
                    config_id,
                    row["context_length_in_tokens"],
                ])

    accuracy = total_correct / total_questions * 100 if total_questions > 0 else 0
    print(f"[DONE] {config_id}: {total_correct}/{total_questions} = {accuracy:.2f}%")
    return output_path, total_correct, total_questions


# ===== CLI =====

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Phase 3: Evaluate QA accuracy with TPPM memory via vLLM")
    parser.add_argument("--config-id", type=str, required=True,
                        help="Config ID from Phase 2 (e.g. 'baseline')")
    parser.add_argument("--no-tppm", action="store_true",
                        help="Run without TPPM memory (ablation baseline)")
    parser.add_argument("--max-questions", type=int, default=None,
                        help="Limit number of questions for smoke testing")
    parser.add_argument("--vllm-url", type=str, default=VLLM_BASE_URL,
                        help="vLLM server URL")
    args = parser.parse_args()

    output_path, correct, total = run_evaluation(
        config_id=args.config_id,
        no_tppm=args.no_tppm,
        max_questions=args.max_questions,
        vllm_url=args.vllm_url,
    )

    print(f"\n[DONE] Results saved to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Verify script parses correctly**

```bash
cd /root/autodl-tmp/wangqihao/Table3-data
python3 -c "import ast; ast.parse(open('scripts/phase3_eval_qa.py').read()); print('Syntax OK')"
```

- [ ] **Step 3: Commit**

```bash
cd /root/autodl-tmp/wangqihao
git add Table3-data/scripts/phase3_eval_qa.py
git commit -m "feat: add Phase 3 vLLM QA evaluation with TPPM context injection

Builds 32K context window (conversation + memory + question), queries
vLLM via OpenAI-compatible API, extracts and scores answers."
```

---

### Task 5: Summarize Results Script

**Files:**
- Create: `/root/autodl-tmp/wangqihao/Table3-data/scripts/summarize.py`

- [ ] **Step 1: Write `summarize.py`**

```python
#!/usr/bin/env python3
"""Summarize Phase 3 evaluation results across all configs.

Outputs accuracy tables (per config, per question_type, per topic) and
generates LaTeX-ready tables for the paper.

Usage:
    python3 summarize.py                           # All configs
    python3 summarize.py --output summary.json     # Custom output path
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

# ===== Paths =====
ROOT = Path("/root/autodl-tmp/wangqihao/Table3-data")
EVAL_DIR = ROOT / "eval_results"
DEFAULT_OUTPUT = ROOT / "eval_summary.json"


def load_results(config_dir: Path) -> list[dict]:
    """Load all result rows from a config's results CSV."""
    results: list[dict] = []
    results_csv = config_dir / "results.csv"
    if not results_csv.exists():
        return results
    with results_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["score"] = row.get("score", "").strip().lower() in ("true", "1", "yes")
            results.append(row)
    return results


def compute_accuracy(rows: list[dict], group_by: str | None = None) -> dict:
    """Compute accuracy, optionally grouped by a column.

    Returns:
        {group_value: {"correct": int, "total": int, "accuracy": float}}
        or {"overall": {...}} if group_by is None.
    """
    if group_by is None:
        correct = sum(1 for r in rows if r["score"])
        total = len(rows)
        return {"overall": {"correct": correct, "total": total,
                            "accuracy": round(correct / total * 100, 2) if total > 0 else 0}}

    groups: dict = defaultdict(lambda: {"correct": 0, "total": 0})
    for row in rows:
        key = row.get(group_by, "unknown")
        groups[key]["total"] += 1
        if row["score"]:
            groups[key]["correct"] += 1

    for key in groups:
        total = groups[key]["total"]
        groups[key]["accuracy"] = round(groups[key]["correct"] / total * 100, 2) if total > 0 else 0

    return dict(groups)


def generate_latex_table(
    summary: dict,
    sweep_name: str,
    configs: list[str],
    by_question_type: bool = False,
) -> str:
    """Generate a LaTeX table for a parameter sweep.

    Args:
        summary: Full summary dict from all configs.
        sweep_name: Display name for the sweep.
        configs: List of config_ids in this sweep.
        by_question_type: If True, break down by question type.

    Returns:
        LaTeX table string.
    """
    if by_question_type:
        # Get all question types from first config
        first = summary.get(configs[0], {}).get("by_question_type", {})
        qtypes = list(first.keys())
        header = " & ".join(["Config"] + qtypes + ["Overall"])
        header += " \\\\\n\\hline"

        rows = []
        for cid in configs:
            cfg = summary.get(cid, {})
            overall = cfg.get("overall", {}).get("accuracy", 0)
            by_type = cfg.get("by_question_type", {})
            vals = [f"{by_type.get(qt, {}).get('accuracy', 0):.1f}" for qt in qtypes]
            row = f"{cid} & " + " & ".join(vals) + f" & {overall:.1f} \\\\"
            rows.append(row)
    else:
        header = "Config & Overall Accuracy \\\\\n\\hline"
        rows = []
        for cid in configs:
            overall = summary.get(cid, {}).get("overall", {}).get("accuracy", 0)
            rows.append(f"{cid} & {overall:.1f}\\% \\\\")

    return (
        "\\begin{table}[ht]\n"
        f"\\caption{{{sweep_name} — QA Accuracy}}\n"
        "\\begin{tabular}{" + "l" + ("c" * (len(header.split("&")) - 1)) + "}\n"
        "\\hline\n"
        + header + "\n"
        + "\n".join(rows) + "\n"
        "\\hline\n"
        "\\end{tabular}\n"
        "\\end{table}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Summarize Phase 3 QA evaluation results")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="Output JSON path")
    args = parser.parse_args()

    # Discover config directories
    config_dirs = sorted(
        d for d in EVAL_DIR.iterdir()
        if d.is_dir() and (d / "results.csv").exists()
    )
    if not config_dirs:
        print("[ERROR] No evaluation results found. Run Phase 3 first.")
        return 1

    # Load and summarize all configs
    summary: dict = {}
    for config_dir in config_dirs:
        config_id = config_dir.name
        rows = load_results(config_dir)
        if not rows:
            continue

        summary[config_id] = {
            "overall": compute_accuracy(rows)["overall"],
            "by_question_type": compute_accuracy(rows, "question_type"),
            "by_topic": compute_accuracy(rows, "topic"),
            "by_persona": compute_accuracy(rows, "persona_id"),
        }

    # Print summary to console
    print(f"\n{'='*70}")
    print(f"Summary — {len(summary)} configs evaluated")
    print(f"{'='*70}")
    print(f"{'Config':<20} {'Correct':>8} {'Total':>6} {'Accuracy':>10}")
    print(f"{'-'*44}")
    for cid in sorted(summary.keys()):
        ov = summary[cid]["overall"]
        print(f"{cid:<20} {ov['correct']:>8} {ov['total']:>6} {ov['accuracy']:>9.2f}%")

    # Save JSON summary
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n[DONE] Summary saved to {args.output}")

    # Generate LaTeX tables for each sweep
    sweep_configs = {
        "Consolidation — write_threshold": ["baseline", "write_0.56", "write_0.62", "write_0.68", "write_0.74", "write_0.80"],
        "Consolidation — promote_threshold": ["baseline", "promote_0.60", "promote_0.66", "promote_0.72", "promote_0.78", "promote_0.84"],
        "Decay λ scale": ["baseline", "decay_0.25x", "decay_0.5x", "decay_1.0x", "decay_2.0x", "decay_4.0x"],
        "Branching — context_threshold": ["baseline", "ctx_0.50", "ctx_0.56", "ctx_0.62", "ctx_0.68", "ctx_0.74"],
    }

    latex_path = ROOT / "eval_summary.tex"
    with latex_path.open("w", encoding="utf-8") as f:
        for sweep_name, configs in sweep_configs.items():
            available = [c for c in configs if c in summary]
            if not available:
                continue
            f.write(generate_latex_table(summary, sweep_name, available))
            f.write("\n\n")
    print(f"[DONE] LaTeX tables saved to {latex_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Verify script parses correctly**

```bash
cd /root/autodl-tmp/wangqihao/Table3-data
python3 -c "import ast; ast.parse(open('scripts/summarize.py').read()); print('Syntax OK')"
```

- [ ] **Step 3: Commit**

```bash
cd /root/autodl-tmp/wangqihao
git add Table3-data/scripts/summarize.py
git commit -m "feat: add summarize.py for aggregating QA results across configs

Generates per-config accuracy, per-question-type breakdown, per-topic,
per-persona, and LaTeX-ready tables for the paper."
```

---

### Task 6: Integration Test — End-to-End Dry Run

- [ ] **Step 1: Verify all scripts import correctly**

```bash
cd /root/autodl-tmp/wangqihao/Table3-data
python3 -c "
import yaml, json, csv, sys
from pathlib import Path
sys.path.insert(0, '/root/autodl-tmp/wangqihao/Mini-Agent-5-1')
from mini_agent.tpm.memory import TemporalProfileMemory, TPMConfig
from mini_agent.tpm.models import ProfileCandidate
print('All imports OK')
print(f'Baseline TPMConfig: write={TPMConfig().write_threshold}, promote={TPMConfig().promote_threshold}')
"
```

- [ ] **Step 2: Verify config YAML parses and resolves correctly**

```bash
cd /root/autodl-tmp/wangqihao/Table3-data
python3 -c "
import yaml
from pathlib import Path
with open('configs/param_sweep.yaml') as f:
    data = yaml.safe_load(f)
sweeps = [k for k in data if k.startswith('sweep_')]
print(f'Found {len(sweeps)} sweeps: {sweeps}')
total_configs = 1  # baseline
for s in sweeps:
    n = len(data[s].get('configs', []))
    print(f'  {s}: {n} configs')
    total_configs += n
print(f'Total configs (incl baseline): {total_configs}')
"
```

Expected output: `Total configs (incl baseline): 21`

- [ ] **Step 3: Verify Phase 2 dry-run with real YAML**

```bash
cd /root/autodl-tmp/wangqihao/Table3-data
python3 scripts/phase2_replay_evolution.py --dry-run
```

Expected: Lists all 21 configs with resolved parameter values.

- [ ] **Step 4: Commit**

```bash
cd /root/autodl-tmp/wangqihao
git add -A Table3-data/
git commit -m "test: verify all Phase 1-3 scripts parse and config resolution works"
```

---

## Execution Order

1. **Task 1** → `configs/param_sweep.yaml` (no dependencies)
2. **Task 2** → `phase1_extract_candidates.py` (requires DeepSeek API key)
3. **Task 3** → `phase2_replay_evolution.py` (requires Phase 1 output)
4. **Task 4** → `phase3_eval_qa.py` (requires Phase 2 output + vLLM server)
5. **Task 5** → `summarize.py` (requires Phase 3 output)
6. **Task 6** → Integration verification (requires all scripts)

## Production Run Commands

```bash
# Phase 1: Extract candidates (one-time, ~10-30 min)
cd /root/autodl-tmp/wangqihao/Table3-data
python3 scripts/phase1_extract_candidates.py

# Phase 2: Replay all configs (pure Python, ~1-5 min per sweep)
python3 scripts/phase2_replay_evolution.py --sweep sweep_2a_write
python3 scripts/phase2_replay_evolution.py --sweep sweep_2a_promote
python3 scripts/phase2_replay_evolution.py --sweep sweep_2b_decay
python3 scripts/phase2_replay_evolution.py --sweep sweep_2c_context

# Phase 3: Evaluate (requires vLLM server, ~30-60 min per config)
# Start vLLM first:
# vllm serve Qwen/Qwen3.5-9B --tensor-parallel-size 2 --max-model-len 32768
python3 scripts/phase3_eval_qa.py --config-id baseline
python3 scripts/phase3_eval_qa.py --config-id baseline --no-tppm  # ablation

# Summarize
python3 scripts/summarize.py
```

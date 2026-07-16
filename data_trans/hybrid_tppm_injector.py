#!/usr/bin/env python3
"""Offline Hybrid TPPM injector for PsyDial-D4.

This script is intentionally standalone. It does not import or modify the
frontend/API memory path. The pipeline is:

1. Read PsyDial-D4 conversations.
2. Truncate each conversation to the first N non-system turns.
3. Ask an LLM to extract and score candidate psychological profile memories.
4. Use explicit Python-side phi scoring to decide admission.
5. Save the filtered memory bank as JSON.

API configuration is explicitly defined in the constants near the top of this
file. Fill EXPLICIT_EXTRACTOR_API_KEY locally before running, or override the
values with --api-base, --api-key and --model.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openai import OpenAI
from tqdm import tqdm


# ===== Hybrid TPPM Hyperparameters =====
ALPHA_1 = 0.25
ALPHA_2 = 0.30
ALPHA_3 = 0.25
ALPHA_4 = 0.20
DELTA_WRITE = 0.68


# ===== Default Paths / Runtime Config =====
DEFAULT_DATASET_PATH = Path("/root/autodl-tmp/wangqihao/datasets/PsyDial/PsyDial-D4/PsyDial-D4.json")
DEFAULT_OUTPUT_PATH = Path("/root/autodl-tmp/wangqihao/data_trans/tppm_memory_bank.json")
DEFAULT_FAILED_PATH = Path("/root/autodl-tmp/wangqihao/data_trans/tppm_failed_sessions.jsonl")
DEFAULT_RAW_RESPONSE_DEBUG_PATH = Path("/root/autodl-tmp/wangqihao/data_trans/tppm_invalid_llm_responses.jsonl")
DEFAULT_LOCAL_MODEL_PATH = Path("/root/autodl-tmp/wangqihao/base_model/models/Qwen2.5-7B-Instruct")

# ===== Explicit API Config =====
# Fill EXPLICIT_EXTRACTOR_API_KEY locally. Do not commit or share real keys.
EXPLICIT_EXTRACTOR_API_BASE = "https://api.deepseek.com"
EXPLICIT_EXTRACTOR_MODEL = "deepseek-v4-flash"
EXPLICIT_EXTRACTOR_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
if not EXPLICIT_EXTRACTOR_API_KEY:
    raise RuntimeError(
        "DEEPSEEK_API_KEY is not set. "
        "Export it before running this script."
    )

DEFAULT_MODEL = EXPLICIT_EXTRACTOR_MODEL
DEFAULT_API_BASE = EXPLICIT_EXTRACTOR_API_BASE
DEFAULT_API_KEY = EXPLICIT_EXTRACTOR_API_KEY
DEFAULT_MAX_TURNS = 15
DEFAULT_MAX_RETRIES = 5
DEFAULT_INITIAL_BACKOFF = 1.0
DEFAULT_MAX_BACKOFF = 30.0
DEFAULT_REQUEST_TIMEOUT = 60.0
DEFAULT_MAX_TOKENS = 1024

VALID_ATTRIBUTES = {"stressor", "affective_state", "coping_style"}


class EmptyLLMResponseError(ValueError):
    """Raised when the API response has no message content."""


class LLMJSONParseError(ValueError):
    """Raised when the LLM response cannot be parsed as JSON."""


class LLMSchemaError(ValueError):
    """Raised when the parsed JSON does not match the required top-level schema."""


@dataclass(slots=True)
class CandidateMemory:
    """Candidate emitted by the LLM feature extractor."""

    attribute: str
    value: str
    evidence: str
    r_score: float
    e_score: float
    u_score: float
    b_score: float
    phi: float


@dataclass(slots=True)
class SessionMemory:
    """Filtered TPPM memory for one PsyDial-D4 session."""

    session_id: str
    source: str
    tppm_memory: list[CandidateMemory]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def clamp_score(value: Any, default: float = 0.0) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        score = default
    return max(0.0, min(1.0, score))


def compute_phi(r_score: float, e_score: float, u_score: float, b_score: float) -> float:
    return (
        ALPHA_1 * r_score
        + ALPHA_2 * e_score
        + ALPHA_3 * u_score
        + ALPHA_4 * b_score
    )


def build_openai_client(
    api_key: str | None = None,
    api_base: str = DEFAULT_API_BASE,
    request_timeout: float = DEFAULT_REQUEST_TIMEOUT,
) -> OpenAI:
    resolved_key = api_key or DEFAULT_API_KEY
    if not resolved_key or resolved_key == "PASTE_YOUR_DEEPSEEK_API_KEY_HERE":
        raise RuntimeError(
            "Missing extractor API key. Fill EXPLICIT_EXTRACTOR_API_KEY in this script "
            "or pass --api-key explicitly."
        )
    return OpenAI(api_key=resolved_key, base_url=api_base, timeout=request_timeout)


def build_system_prompt() -> str:
    return """
You are a psychological profile memory extractor for a Hybrid TPPM system.

Your only job is feature extraction and scoring. You MUST NOT decide whether a memory should be written.
Python code will make the final write decision with a fixed mathematical formula.

Read the truncated mental-health support dialogue and extract candidate PPMUs for these attributes only:
- stressor: the user's core pressure source or triggering situation.
- affective_state: the user's emotional state or mood pattern.
- coping_style: how the user tends to respond, cope, avoid, suppress, seek help, ruminate, rationalize, or regulate emotions.

For every candidate, output four scores in [0.0, 1.0]:
- r_score: relevance to psychological profile understanding.
- e_score: explicitness of evidence in the dialogue.
- u_score: utility for future psychological support and personalization.
- b_score: tendency to persist beyond a single fleeting utterance.

Important constraints:
1. Output exactly one JSON object and nothing else.
2. Do not output Markdown, explanations, comments, or trailing commas.
3. All JSON keys and strings must use double quotes.
4. Do not copy long raw dialogue. Keep evidence short and privacy-minimized.
5. Do not invent facts that are not supported by the dialogue.
6. Do not output clinical diagnosis labels.
7. Ignore therapist/assistant suggestions unless the user accepts or describes them.
8. If no useful candidate exists, return exactly {"candidates":[]}.

Required JSON schema:
{"candidates":[{"attribute":"stressor|affective_state|coping_style","value":"short memory value","evidence":"short supporting reason","r_score":0.0,"e_score":0.0,"u_score":0.0,"b_score":0.0}]}
""".strip()


def append_invalid_llm_response(
    *,
    session_id: str,
    model: str,
    attempt: int,
    response_id: str | None,
    usage: Any,
    finish_reason: str | None,
    content: str,
    dialogue_length: int,
    max_tokens: int,
    error: Exception,
) -> None:
    DEFAULT_RAW_RESPONSE_DEBUG_PATH.parent.mkdir(parents=True, exist_ok=True)
    stripped = clean_json_text(content)
    usage_payload = usage
    if hasattr(usage, "model_dump"):
        usage_payload = usage.model_dump()
    elif hasattr(usage, "dict"):
        usage_payload = usage.dict()

    record = {
        "timestamp": utc_now_iso(),
        "session_id": session_id,
        "model": model,
        "attempt": attempt,
        "response_id": response_id,
        "usage": usage_payload,
        "finish_reason": finish_reason,
        "dialogue_length": dialogue_length,
        "max_tokens": max_tokens,
        "error": repr(error),
        "error_type": type(error).__name__,
        "raw_content": content,
        "stripped_content": stripped,
        "content_preview": content[:4000],
    }
    with DEFAULT_RAW_RESPONSE_DEBUG_PATH.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def clean_json_text(content: str) -> str:
    stripped = (content or "").lstrip("\ufeff").strip()
    fence_match = re.fullmatch(r"```(?:json|JSON)?\s*(.*?)\s*```", stripped, flags=re.DOTALL)
    if fence_match:
        stripped = fence_match.group(1).strip()
    return stripped


def remove_trailing_commas(text: str) -> str:
    return re.sub(r",\s*([}\]])", r"\1", text)


def first_json_object(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            payload, _end = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise LLMJSONParseError("No complete JSON object found in LLM output.")


def looks_like_unclosed_json(text: str) -> bool:
    if not text.lstrip().startswith("{"):
        return False
    in_string = False
    escaped = False
    stack: list[str] = []
    for char in text:
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char in "{[":
            stack.append(char)
        elif char == "}":
            if stack and stack[-1] == "{":
                stack.pop()
        elif char == "]":
            if stack and stack[-1] == "[":
                stack.pop()
    return bool(stack or in_string)


def parse_llm_json(content: str) -> dict[str, Any]:
    stripped = clean_json_text(content)
    if not stripped:
        raise EmptyLLMResponseError("LLM returned empty content.")

    parse_errors: list[str] = []
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError as error:
        parse_errors.append(repr(error))
        repaired = remove_trailing_commas(stripped)
        if repaired != stripped:
            try:
                payload = json.loads(repaired)
            except json.JSONDecodeError as repaired_error:
                parse_errors.append(repr(repaired_error))
                payload = None
        else:
            payload = None

        if payload is None:
            try:
                payload = first_json_object(repaired if "repaired" in locals() else stripped)
            except Exception as object_error:
                parse_errors.append(repr(object_error))
                preview = stripped[:500].replace("\n", "\\n")
                raise LLMJSONParseError(
                    f"Could not parse LLM output as JSON. preview={preview!r}; errors={parse_errors}"
                ) from error

    if not isinstance(payload, dict):
        raise LLMSchemaError("LLM JSON output must be an object.")
    return payload


def validate_top_level_payload(payload: dict[str, Any]) -> list[Any]:
    if not isinstance(payload, dict):
        raise LLMSchemaError("Top-level LLM JSON must be an object.")
    if "candidates" not in payload:
        raise LLMSchemaError("Top-level LLM JSON must contain 'candidates'.")
    candidates = payload["candidates"]
    if not isinstance(candidates, list):
        raise LLMSchemaError("Top-level field 'candidates' must be a list.")
    return candidates


def get_llm_candidates(
    dialogue_text: str,
    *,
    session_id: str,
    client: OpenAI | None = None,
    model: str = DEFAULT_MODEL,
    max_retries: int = DEFAULT_MAX_RETRIES,
    initial_backoff: float = DEFAULT_INITIAL_BACKOFF,
    max_backoff: float = DEFAULT_MAX_BACKOFF,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> list[dict[str, Any]]:
    """Call an OpenAI-compatible API to extract scored PPMU candidates."""

    if not dialogue_text.strip():
        return []

    llm_client = client or build_openai_client()
    system_prompt = build_system_prompt()
    user_prompt = (
        "Extract scored TPPM candidate memories from the following truncated dialogue.\n"
        "Remember: output JSON only.\n\n"
        f"{dialogue_text}"
    )

    last_error: Exception | None = None
    max_token_cap = max(max_tokens, 4096)
    for attempt in range(1, max_retries + 1):
        attempt_max_tokens = min(max_tokens * (2 ** (attempt - 1)), max_token_cap)
        try:
            response = llm_client.chat.completions.create(
                model=model,
                temperature=0,
                max_tokens=attempt_max_tokens,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            choice = response.choices[0]
            finish_reason = getattr(choice, "finish_reason", None)
            response_id = getattr(response, "id", None)
            usage = getattr(response, "usage", None)
            content = choice.message.content or ""
            if not content.strip():
                error = EmptyLLMResponseError(
                    f"LLM returned empty content; finish_reason={finish_reason}, "
                    f"response_id={response_id}, usage={usage}"
                )
                append_invalid_llm_response(
                    session_id=session_id,
                    model=model,
                    attempt=attempt,
                    response_id=response_id,
                    usage=usage,
                    finish_reason=finish_reason,
                    content=content,
                    dialogue_length=len(dialogue_text),
                    max_tokens=attempt_max_tokens,
                    error=error,
                )
                raise error

            if finish_reason == "length":
                error = ValueError(f"LLM output was truncated by max_tokens={attempt_max_tokens}.")
                append_invalid_llm_response(
                    session_id=session_id,
                    model=model,
                    attempt=attempt,
                    response_id=response_id,
                    usage=usage,
                    finish_reason=finish_reason,
                    content=content,
                    dialogue_length=len(dialogue_text),
                    max_tokens=attempt_max_tokens,
                    error=error,
                )
                raise error

            try:
                payload = parse_llm_json(content)
            except Exception as parse_error:
                append_invalid_llm_response(
                    session_id=session_id,
                    model=model,
                    attempt=attempt,
                    response_id=response_id,
                    usage=usage,
                    finish_reason=finish_reason,
                    content=content,
                    dialogue_length=len(dialogue_text),
                    max_tokens=attempt_max_tokens,
                    error=parse_error,
                )
                raise

            try:
                candidates = validate_top_level_payload(payload)
            except Exception as schema_error:
                append_invalid_llm_response(
                    session_id=session_id,
                    model=model,
                    attempt=attempt,
                    response_id=response_id,
                    usage=usage,
                    finish_reason=finish_reason,
                    content=content,
                    dialogue_length=len(dialogue_text),
                    max_tokens=attempt_max_tokens,
                    error=schema_error,
                )
                raise
            return candidates
        except Exception as exc:
            last_error = exc
            if attempt >= max_retries:
                break
            sleep_seconds = min(max_backoff, initial_backoff * (2 ** (attempt - 1)))
            sleep_seconds += random.uniform(0.0, 0.25 * sleep_seconds)
            print(
                f"[WARN] LLM call failed on attempt {attempt}/{max_retries} "
                f"(max_tokens={attempt_max_tokens}): {exc}. "
                f"Retrying in {sleep_seconds:.1f}s...",
                file=sys.stderr,
            )
            time.sleep(sleep_seconds)

    raise RuntimeError(f"LLM extraction failed after {max_retries} attempts: {last_error}")


def score_field(raw: dict[str, Any], field: str) -> float:
    if field not in raw:
        raise ValueError(f"missing score field {field}")
    value = raw[field]
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric, got {value!r}") from exc
    if not 0.0 <= numeric <= 1.0:
        raise ValueError(f"{field} must be in [0,1], got {numeric!r}")
    return numeric


def normalize_candidate(raw: dict[str, Any]) -> tuple[CandidateMemory | None, str | None]:
    """Validate one raw LLM candidate and attach Python-side phi."""

    if not isinstance(raw, dict):
        return None, "candidate is not an object"

    attribute = str(raw.get("attribute", "")).strip().lower()
    if attribute not in VALID_ATTRIBUTES:
        return None, f"invalid attribute {attribute!r}"

    value_raw = raw.get("value")
    if not isinstance(value_raw, str) or not value_raw.strip():
        return None, "value must be a non-empty string"
    value = value_raw.strip()

    evidence_raw = raw.get("evidence", "")
    if not isinstance(evidence_raw, str):
        return None, "evidence must be a string"
    evidence = evidence_raw.strip()

    try:
        r_score = score_field(raw, "r_score")
        e_score = score_field(raw, "e_score")
        u_score = score_field(raw, "u_score")
        b_score = score_field(raw, "b_score")
    except ValueError as exc:
        return None, str(exc)
    phi = compute_phi(r_score, e_score, u_score, b_score)

    return CandidateMemory(
        attribute=attribute,
        value=value,
        evidence=evidence,
        r_score=r_score,
        e_score=e_score,
        u_score=u_score,
        b_score=b_score,
        phi=round(phi, 6),
    ), None


def process_session(
    session_id: str,
    dialogue_text: str,
    *,
    client: OpenAI | None = None,
    model: str = DEFAULT_MODEL,
    max_retries: int = DEFAULT_MAX_RETRIES,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> SessionMemory:
    """Run Hybrid TPPM extraction and explicit Python filtering for one session."""

    raw_candidates = get_llm_candidates(
        dialogue_text,
        session_id=session_id,
        client=client,
        model=model,
        max_retries=max_retries,
        max_tokens=max_tokens,
    )

    accepted: list[CandidateMemory] = []
    for raw in raw_candidates:
        candidate, drop_reason = normalize_candidate(raw)
        if candidate is None:
            print(
                f"[DROP] session={session_id} invalid candidate: reason={drop_reason}; raw={raw}",
                file=sys.stderr,
            )
            continue

        if candidate.phi > DELTA_WRITE:
            accepted.append(candidate)
        else:
            print(
                "[DROP] "
                f"session={session_id} attribute={candidate.attribute} "
                f"phi={candidate.phi:.4f} <= DELTA_WRITE={DELTA_WRITE:.2f} "
                f"value={candidate.value[:80]}",
                file=sys.stderr,
            )

    return SessionMemory(
        session_id=session_id,
        source="PsyDial-D4",
        tppm_memory=accepted,
    )


def load_dataset(path: Path) -> list[dict[str, Any]]:
    """Load PsyDial-D4 and normalize common top-level wrappers to a list."""

    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]

    if isinstance(payload, dict):
        if isinstance(payload.get("messages"), list):
            return [payload]
        for key in ("data", "conversations", "sessions", "items"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]

    raise ValueError(f"Unsupported dataset structure in {path}")


def session_id_for(index: int, item: dict[str, Any]) -> str:
    for key in ("session_id", "id", "dialogue_id", "conversation_id"):
        value = item.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return f"psydial_d4_{index:05d}"


def normalized_non_system_messages(item: dict[str, Any]) -> list[dict[str, str]]:
    messages = item.get("messages")
    if not isinstance(messages, list):
        return []

    normalized: list[dict[str, str]] = []
    for message in messages:
        if not isinstance(message, dict):
            continue

        role = str(message.get("role", "")).strip().lower()
        content = message.get("content", "")
        if role == "system":
            continue
        if role not in {"user", "assistant"}:
            continue

        if isinstance(content, list):
            content = "\n".join(
                str(block.get("text", block)) if isinstance(block, dict) else str(block)
                for block in content
            )
        content_text = str(content).strip()
        if not content_text:
            continue

        normalized.append({"role": role, "content": content_text})
    return normalized


def selected_turn_count(total_turns: int, max_turns: int, turn_ratio: float | None = None) -> int:
    if total_turns <= 0:
        return 0
    if turn_ratio is None:
        return min(total_turns, max_turns)
    if not 0.0 < turn_ratio <= 1.0:
        raise ValueError("--turn-ratio must be in (0, 1].")
    ratio_turns = max(1, math.ceil(total_turns * turn_ratio))
    return min(total_turns, max_turns, ratio_turns)


def format_truncated_dialogue(
    item: dict[str, Any],
    max_turns: int = DEFAULT_MAX_TURNS,
    turn_ratio: float | None = None,
) -> str:
    """Format the first turns selected by fixed max_turns or a session ratio."""

    messages = normalized_non_system_messages(item)
    take_turns = selected_turn_count(len(messages), max_turns=max_turns, turn_ratio=turn_ratio)

    formatted: list[str] = []
    for message in messages[:take_turns]:
        role = message["role"]
        speaker = "User" if role == "user" else "Assistant"
        formatted.append(f"{speaker}: {message['content']}")

    return "\n".join(formatted)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def append_failed_session(
    path: Path,
    session_id: str,
    error: Exception,
    *,
    dialogue_length: int,
    model: str,
    max_retries: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "session_id": session_id,
        "error": repr(error),
        "error_type": type(error).__name__,
        "timestamp": utc_now_iso(),
        "dialogue_length": dialogue_length,
        "model": model,
        "max_retries": max_retries,
        "final_attempt_number": max_retries,
        "invalid_response_debug_file": str(DEFAULT_RAW_RESPONSE_DEBUG_PATH),
    }
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_memory_bank_payload(sessions: list[SessionMemory], dataset_path: Path, model: str) -> dict[str, Any]:
    total_memories = sum(len(session.tppm_memory) for session in sessions)
    return {
        "metadata": {
            "source_dataset": str(dataset_path),
            "generated_at": utc_now_iso(),
            "extractor_model": model,
            "design": "Hybrid TPPM: LLM feature scoring + Python explicit phi filtering",
            "alphas": {
                "ALPHA_1": ALPHA_1,
                "ALPHA_2": ALPHA_2,
                "ALPHA_3": ALPHA_3,
                "ALPHA_4": ALPHA_4,
            },
            "delta_write": DELTA_WRITE,
            "accepted_session_count": len(sessions),
            "accepted_memory_count": total_memories,
        },
        "sessions": [
            {
                "session_id": session.session_id,
                "source": session.source,
                "tppm_memory": [asdict(candidate) for candidate in session.tppm_memory],
            }
            for session in sessions
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline Hybrid TPPM injector for PsyDial-D4.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH, help="Path to PsyDial-D4.json")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="Output memory bank JSON path")
    parser.add_argument(
        "--failed-output",
        type=Path,
        default=DEFAULT_FAILED_PATH,
        help="JSONL path for sessions that fail after retries",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="OpenAI-compatible extractor model name, e.g. deepseek-chat or provider-specific aliases.",
    )
    parser.add_argument("--api-base", default=DEFAULT_API_BASE, help="OpenAI-compatible extractor API base")
    parser.add_argument("--api-key", default=DEFAULT_API_KEY, help="Extractor API key")
    parser.add_argument("--max-turns", type=int, default=DEFAULT_MAX_TURNS, help="First N non-system turns per session")
    parser.add_argument(
        "--turn-ratio",
        type=float,
        default=None,
        help="Optional ratio of each session's non-system turns to use, e.g. 0.35 for the first 35%%. Still capped by --max-turns.",
    )
    parser.add_argument("--max-sessions", type=int, default=None, help="Optional cap for smoke tests")
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES, help="API retry count per session")
    parser.add_argument("--request-timeout", type=float, default=DEFAULT_REQUEST_TIMEOUT, help="HTTP request timeout in seconds")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS, help="Maximum completion tokens for extractor JSON")
    parser.add_argument(
        "--skip-empty",
        action="store_true",
        help="Do not include sessions with zero accepted memories in final output",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    dataset = load_dataset(args.dataset)
    if args.max_sessions is not None:
        dataset = dataset[: args.max_sessions]

    client = build_openai_client(
        api_key=args.api_key,
        api_base=args.api_base,
        request_timeout=args.request_timeout,
    )
    accepted_sessions: list[SessionMemory] = []
    failed_count = 0
    empty_dialogue_count = 0

    print(f"[INFO] dataset={args.dataset}")
    print(f"[INFO] sessions={len(dataset)}")
    print(f"[INFO] model={args.model}")
    print(f"[INFO] api_base={args.api_base}")
    print(f"[INFO] request_timeout={args.request_timeout}")
    print(f"[INFO] max_tokens={args.max_tokens}")
    print(f"[INFO] max_turns={args.max_turns}")
    print(f"[INFO] turn_ratio={args.turn_ratio}")
    print(f"[INFO] output={args.output}")
    print(
        "[INFO] phi="
        f"{ALPHA_1}*r + {ALPHA_2}*e + {ALPHA_3}*u + {ALPHA_4}*b, "
        f"DELTA_WRITE={DELTA_WRITE}"
    )

    for index, item in enumerate(tqdm(dataset, desc="Injecting PsyDial-D4")):
        session_id = session_id_for(index, item)
        dialogue_text = format_truncated_dialogue(
            item,
            max_turns=args.max_turns,
            turn_ratio=args.turn_ratio,
        )
        if not dialogue_text:
            empty_dialogue_count += 1
            print(f"[DROP] session={session_id} empty dialogue after truncation", file=sys.stderr)
            continue

        try:
            session_memory = process_session(
                session_id,
                dialogue_text,
                client=client,
                model=args.model,
                max_retries=args.max_retries,
                max_tokens=args.max_tokens,
            )
        except Exception as exc:
            failed_count += 1
            append_failed_session(
                args.failed_output,
                session_id,
                exc,
                dialogue_length=len(dialogue_text),
                model=args.model,
                max_retries=args.max_retries,
            )
            print(f"[ERROR] session={session_id} failed: {exc}", file=sys.stderr)
            continue

        if session_memory.tppm_memory or not args.skip_empty:
            accepted_sessions.append(session_memory)

    payload = build_memory_bank_payload(accepted_sessions, args.dataset, args.model)
    payload["metadata"]["failed_session_count"] = failed_count
    payload["metadata"]["empty_dialogue_count"] = empty_dialogue_count
    payload["metadata"]["input_session_count"] = len(dataset)
    payload["metadata"]["max_turns"] = args.max_turns
    payload["metadata"]["turn_ratio"] = args.turn_ratio
    payload["metadata"]["skip_empty"] = args.skip_empty

    write_json(args.output, payload)
    print(f"[DONE] wrote memory bank: {args.output}")
    print(f"[DONE] accepted sessions in output: {len(accepted_sessions)}")
    print(f"[DONE] accepted memories: {payload['metadata']['accepted_memory_count']}")
    print(f"[DONE] failed sessions: {failed_count}")
    if failed_count:
        print(f"[DONE] failed session log: {args.failed_output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

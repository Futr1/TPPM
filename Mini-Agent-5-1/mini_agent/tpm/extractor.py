"""Candidate extraction for Temporal Profile Memory."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

try:
    import requests
except ModuleNotFoundError:  # pragma: no cover - environment dependent
    requests = None

from .models import (
    MEMORY_TYPE_VALUES,
    SLOT_VALUES,
    ProfileCandidate,
    default_memory_type,
    migrate_profile_type,
)


class ProfileExtractor:
    """Candidate extractor interface."""

    def extract(self, text: str, scene: str = "general") -> list[ProfileCandidate]:
        raise NotImplementedError


@dataclass(slots=True)
class RegexProfileExtractor(ProfileExtractor):
    """Heuristic extractor used to bootstrap TPM on top of Mini-Agent."""

    def extract(self, text: str, scene: str = "general") -> list[ProfileCandidate]:
        candidates: list[ProfileCandidate] = []
        specs = [
            (r"\bmy name is ([^.,;!?]+)", "identity", "support", "trait", 0.96, 0.95),
            (r"\bI am (?:a|an) ([^.,;!?]+)", "identity", "support", "trait", 0.82, 0.86),
            (r"\bI like ([^.,;!?]+)", "interest", "behavior", "trait", 0.78, 0.72),
            (r"\bI love ([^.,;!?]+)", "interest", "behavior", "trait", 0.86, 0.78),
            (r"\bI prefer ([^.,;!?]+)", "style", "cognitive", "trait", 0.87, 0.82),
            (r"\bI work (?:on|with) ([^.,;!?]+)", "project_focus", "behavior", "coping", 0.82, 0.68),
            (r"\bI (?:need|want) ([^.,;!?]+)", "current_goal", "behavior", "coping", 0.8, 0.54),
            (r"\bfor this project[, ]+([^.,;!?]+)", "project_constraint", "behavior", "coping", 0.76, 0.62),
        ]

        for pattern, attribute, slot, memory_type, confidence, stability in specs:
            for match in re.finditer(pattern, text, flags=re.IGNORECASE):
                value = self._clean_value(match.group(1)) if match.groups() else match.group(0).strip()
                if not value:
                    continue
                candidates.append(
                    ProfileCandidate(
                        attribute=attribute,
                        value=value,
                        context=text,
                        slot=slot,
                        memory_type=memory_type,
                        scene=scene,
                        confidence=confidence,
                        stability=stability,
                        explicitness=0.92,
                        relevance=1.0,
                        utility=0.9,
                    )
                )

        return candidates

    @staticmethod
    def _clean_value(value: str) -> str:
        return value.strip().strip("\"'` ").rstrip(".")


@dataclass(slots=True)
class LLMProfileExtractor(ProfileExtractor):
    """LLM-powered extractor that asks Qwen to emit structured profile candidates."""

    api_key: str
    api_base: str
    model: str
    timeout: float = 30.0
    max_candidates: int = 8
    fallback_extractor: ProfileExtractor | None = None

    def extract(self, text: str, scene: str = "general") -> list[ProfileCandidate]:
        if requests is None or not self.api_key:
            return self._fallback(text, scene)

        try:
            payload = self._build_payload(text=text, scene=scene)
            response = requests.post(
                self._chat_completions_url(),
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            content = self._extract_response_content(response.json())
            candidates = self._parse_candidates(content=content, original_text=text, scene=scene)
            if candidates:
                return candidates[: self.max_candidates]
        except Exception:
            return self._fallback(text, scene)

        return self._fallback(text, scene)

    def _build_payload(self, text: str, scene: str) -> dict[str, Any]:
        schema_hint = {
            "candidates": [
                {
                    "attribute": "short_attribute_name",
                    "value": "profile_value",
                    "context": "supporting_span_or_short_reason",
                    "slot": "affect|stressor|cognitive|coping|support|behavior|risk",
                    "memory_type": "affect|stressor|coping|support|trait",
                    "scene": scene,
                    "confidence": 0.0,
                    "stability": 0.0,
                    "relevance": 1.0,
                    "explicitness": 0.0,
                    "utility": 0.0,
                    "source": "llm_qwen",
                }
            ]
        }
        system_prompt = (
            "You are a profile candidate extractor for Temporal Profile Memory (TPM). "
            "Extract stable, reusable, and scene-conditioned user profile information from the latest user utterance. "
            "Return ONLY valid JSON, no markdown, no explanation."
        )
        user_prompt = (
            "Task: extract profile candidates for TPM.\n"
            f"Current scene: {scene}\n"
            f"Latest user utterance:\n{text}\n\n"
            "Extraction rules:\n"
            "1. Keep only user-related profile facts, preferences, goals, style tendencies, identity, or stable project context.\n"
            "2. Ignore assistant behavior, transient tool output requests, and generic conversational filler.\n"
            "3. Use concise attribute names like identity, interest, current_goal, style, project_focus, preference.\n"
            "4. slot must be one of: affect, stressor, cognitive, coping, support, behavior, risk.\n"
            "5. memory_type must be one of: affect, stressor, coping, support, trait.\n"
            "6. confidence, stability, relevance, explicitness, utility must be numbers in [0,1].\n"
            "7. utility measures how central this fact is to the user's enduring profile.\n"
            "8. Prefer higher stability for repeated or enduring traits; lower stability for short-term goals.\n"
            "9. If there is no useful profile memory candidate, return {\"candidates\": []}.\n\n"
            f"Output JSON schema example:\n{json.dumps(schema_hint, ensure_ascii=False)}"
        )
        return {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0,
        }

    def _chat_completions_url(self) -> str:
        return self.api_base.rstrip("/") + "/chat/completions"

    def _extract_response_content(self, payload: dict[str, Any]) -> str:
        choices = payload.get("choices") or []
        if not choices:
            return ""
        message = choices[0].get("message") or {}
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
            return "\n".join(text_parts)
        return ""

    def _parse_candidates(self, content: str, original_text: str, scene: str) -> list[ProfileCandidate]:
        parsed = self._extract_json(content)
        if isinstance(parsed, dict):
            raw_candidates = parsed.get("candidates", [])
        elif isinstance(parsed, list):
            raw_candidates = parsed
        else:
            raw_candidates = []

        candidates: list[ProfileCandidate] = []
        for item in raw_candidates:
            if not isinstance(item, dict):
                continue
            attribute = str(item.get("attribute", "")).strip()
            value = str(item.get("value", "")).strip()
            if not attribute or not value:
                continue

            slot = str(item.get("slot", "")).strip().lower()
            memory_type = str(item.get("memory_type", "")).strip().lower()
            if not slot and item.get("profile_type"):
                slot, memory_type = migrate_profile_type(str(item["profile_type"]))
            if slot not in SLOT_VALUES:
                slot = "coping"
            if memory_type not in MEMORY_TYPE_VALUES:
                memory_type = default_memory_type(slot)

            candidates.append(
                ProfileCandidate(
                    attribute=attribute,
                    value=value,
                    context=str(item.get("context") or original_text).strip() or original_text,
                    slot=slot,
                    memory_type=memory_type,
                    scene=str(item.get("scene") or scene).strip() or scene,
                    confidence=self._clamp(item.get("confidence"), default=0.72),
                    stability=self._clamp(item.get("stability"), default=self._default_stability(slot)),
                    relevance=self._clamp(item.get("relevance", item.get("recency")), default=1.0),
                    explicitness=self._clamp(item.get("explicitness"), default=0.8),
                    utility=self._clamp(item.get("utility", item.get("user_relevance")), default=0.82),
                    source=str(item.get("source") or "llm_qwen").strip() or "llm_qwen",
                )
            )

        return candidates

    def _extract_json(self, content: str) -> Any:
        stripped = content.strip()
        if stripped.startswith("```"):
            stripped = re.sub(r"^```(?:json)?", "", stripped, flags=re.IGNORECASE).strip()
            stripped = re.sub(r"```$", "", stripped).strip()

        try:
            return json.loads(stripped)
        except Exception:
            pass

        first_obj = stripped.find("{")
        last_obj = stripped.rfind("}")
        if first_obj != -1 and last_obj != -1 and last_obj > first_obj:
            try:
                return json.loads(stripped[first_obj : last_obj + 1])
            except Exception:
                pass

        first_arr = stripped.find("[")
        last_arr = stripped.rfind("]")
        if first_arr != -1 and last_arr != -1 and last_arr > first_arr:
            return json.loads(stripped[first_arr : last_arr + 1])

        raise ValueError("No valid JSON found in LLM extractor output.")

    def _fallback(self, text: str, scene: str) -> list[ProfileCandidate]:
        if self.fallback_extractor is None:
            return []
        return self.fallback_extractor.extract(text=text, scene=scene)

    @staticmethod
    def _clamp(value: Any, default: float) -> float:
        try:
            numeric = float(value)
        except Exception:
            numeric = default
        return max(0.0, min(1.0, numeric))

    @staticmethod
    def _default_stability(slot: str) -> float:
        defaults = {
            "affect": 0.5,
            "stressor": 0.56,
            "cognitive": 0.82,
            "coping": 0.7,
            "support": 0.75,
            "behavior": 0.68,
            "risk": 0.6,
        }
        return defaults.get(slot, 0.6)

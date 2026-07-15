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
            # --- 中文心理/风险信号 ---
            (r"自伤|割腕|不想活|想死|自杀|轻生", "self_harm_risk", "risk", "affect", 0.95, 0.6),
            (r"焦虑|焦虑症|恐慌", "anxiety", "affect", "affect", 0.85, 0.55),
            (r"抑郁|很丧|没(?:有)?动力|提不起劲", "depression", "affect", "affect", 0.82, 0.55),
            (r"压力(?:很|太)?大|喘不过气", "stress", "stressor", "stressor", 0.84, 0.58),
            (r"失眠|睡不着|睡不好|多梦", "sleep", "behavior", "affect", 0.8, 0.6),
            (r"我(?:叫|的名字是)\s*([^.,;!?]+)", "identity", "support", "trait", 0.9, 0.9),
            (r"我(?:喜欢|爱好)\s*([^.,;!?]+)", "interest", "behavior", "trait", 0.78, 0.7),
            (r"我(?:倾向|偏好)\s*([^.,;!?]+)", "style", "cognitive", "trait", 0.8, 0.78),
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
                    "source": "llm_extractor",
                }
            ]
        }
        system_prompt = (
            "你是时间心理画像记忆（TPM）的候选抽取器，服务于长期心理健康支持。"
            "从用户最新发言中抽取稳定、可复用、情境相关的心理画像信息。"
            "只输出合法 JSON，不要 markdown，不要解释。"
        )
        user_prompt = (
            "任务：为 TPM 抽取心理画像候选。\n"
            f"当前场景：{scene}\n"
            f"最新用户发言：\n{text}\n\n"
            "抽取规则：\n"
            "1. slot 必须是 7 个心理子空间之一：affect(情绪)/stressor(压力源)/cognitive(认知信念)/"
            "coping(应对方式)/support(社会支持)/behavior(行为节律)/risk(风险信号，如自伤/自杀念头)。\n"
            "2. memory_type 必须是 5 个时间–安全类型之一：affect/stressor/coping/support/trait；"
            "若不确定，可只给 slot，系统会按默认回填 memory_type。\n"
            "3. 只保留与用户心理画像相关的事实：情绪状态、压力源、应对方式、社会支持、行为节律、稳定信念、风险信号。\n"
            "4. 忽略助手行为、工具输出请求、无意义寒暄。\n"
            "5. relevance/utility/explicitness/confidence/stability 取 [0,1] 数值；"
            "relevance 衡量与用户持久画像的相关性，utility 衡量对支持的实用性。\n"
            "6. risk 信号（自伤/自杀等）必须置 slot=risk 并提高 confidence。\n"
            "7. 无可用候选时返回 {\"candidates\": []}。\n\n"
            f"输出 JSON schema 示例：\n{json.dumps(schema_hint, ensure_ascii=False)}"
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

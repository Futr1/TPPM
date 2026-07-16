"""Extractor API key resolution (extracted from Mini-Agent CLI)."""

import os


def _extractor_api_key(memory_extractor_cfg, llm_config) -> str | None:
    """Resolve extractor API key: explicit config > DEEPSEEK_API_KEY env var > main LLM key."""
    primary = getattr(memory_extractor_cfg, "api_key", None)
    if primary:
        return primary
    return os.environ.get("DEEPSEEK_API_KEY") or getattr(llm_config, "api_key", None)

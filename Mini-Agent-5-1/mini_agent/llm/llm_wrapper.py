"""LLM client wrapper that supports multiple providers.

This module provides a unified interface for different LLM providers
(Anthropic, OpenAI, DeepSeek, GLM) through a single LLMClient class.
"""

import logging

from ..retry import RetryConfig
from ..schema import LLMProvider, LLMResponse, Message
from .base import LLMClientBase

logger = logging.getLogger(__name__)


class LLMClient:
    """LLM Client wrapper supporting multiple providers.

    This class provides a unified interface for different LLM providers.
    It automatically instantiates the correct underlying client based on
    the provider parameter.

    For MiniMax API (api.minimax.io or api.minimaxi.com), it appends the
    appropriate endpoint suffix based on provider:
    - anthropic: /anthropic
    - openai/deepseek/glm: /v1

    For local or third-party APIs, it uses the api_base as-is.
    """

    # MiniMax API domains that need automatic suffix handling
    MINIMAX_DOMAINS = ("api.minimax.io", "api.minimaxi.com")

    def __init__(
        self,
        api_key: str,
        provider: str,
        api_base: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        model: str = "qwen2.5-7b-instruct",
        retry_config: RetryConfig | None = None,
        **kwargs,
    ):
        """Initialize LLM client with specified provider.

        Args:
            api_key: API key for authentication
            provider: LLM provider (anthropic/openai/deepseek/glm)
            api_base: Base URL for the API (default: local OpenAI-compatible endpoint)
                     For MiniMax API, suffix is auto-appended based on provider.
                     For third-party or local APIs, used as-is.
            model: Model name to use
            retry_config: Optional retry configuration
        """
        self.provider = provider
        self.api_key = api_key
        self.model = model
        self.retry_config = retry_config or RetryConfig()

        # Normalize api_base (remove trailing slash)
        api_base = api_base.rstrip("/")

        # Check if this is a MiniMax API endpoint
        is_minimax = any(domain in api_base for domain in self.MINIMAX_DOMAINS)

        if is_minimax:
            # For MiniMax API, ensure correct suffix based on provider
            # Strip any existing suffix first
            api_base = api_base.replace("/anthropic", "").replace("/v1", "")
            if provider == LLMProvider.ANTHROPIC:
                full_api_base = f"{api_base}/anthropic"
            elif provider in (LLMProvider.OPENAI, LLMProvider.DEEPSEEK, LLMProvider.GLM):
                full_api_base = f"{api_base}/v1"
            else:
                raise ValueError(f"Unsupported provider: {provider}")
        else:
            # For third-party APIs, use api_base as-is
            full_api_base = api_base

        self.api_base = full_api_base

        # Instantiate the appropriate client
        self._client: LLMClientBase
        if provider == LLMProvider.ANTHROPIC:
            try:
                from .anthropic_client import AnthropicClient
            except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
                raise ModuleNotFoundError(
                    "Anthropic provider requires the 'anthropic' package to be installed."
                ) from exc

            self._client = AnthropicClient(
                api_key=api_key,
                api_base=full_api_base,
                model=model,
                retry_config=retry_config,
            )
        elif provider in (LLMProvider.OPENAI, LLMProvider.DEEPSEEK, LLMProvider.GLM):
            from .openai_client import OpenAIClient

            self._client = OpenAIClient(
                api_key=api_key,
                api_base=full_api_base,
                model=model,
                retry_config=retry_config,
            )
        # note: local_lora provider removed (LoRA abandoned)
        else:
            raise ValueError(f"Unsupported provider: {provider}")

        logger.info("Initialized LLM client with provider: %s, api_base: %s", provider, full_api_base)

    @property
    def retry_callback(self):
        """Get retry callback."""
        return self._client.retry_callback

    @retry_callback.setter
    def retry_callback(self, value):
        """Set retry callback."""
        self._client.retry_callback = value

    async def generate(
        self,
        messages: list[Message],
        tools: list | None = None,
    ) -> LLMResponse:
        """Generate response from LLM.

        Args:
            messages: List of conversation messages
            tools: Optional list of Tool objects or dicts

        Returns:
            LLMResponse containing the generated content
        """
        return await self._client.generate(messages, tools)

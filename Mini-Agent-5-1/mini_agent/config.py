"""Configuration management module

Provides unified configuration loading and management functionality
"""

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class RetryConfig(BaseModel):
    """Retry configuration"""

    enabled: bool = True
    max_retries: int = 3
    initial_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0


class LLMConfig(BaseModel):
    """LLM configuration"""

    api_key: str
    api_base: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    model: str = "qwen2.5-7b-instruct"
    provider: str = "openai"  # "anthropic", "openai", "deepseek", "glm", or "local_lora"
    lora_adapter_dir: str | None = None
    lora_user_id: str | None = None
    retry: RetryConfig = Field(default_factory=RetryConfig)
    # Optional: use a separate OpenAI-compatible endpoint (e.g., local vLLM wrapper)
    openai_compat_enabled: bool = False
    openai_compat_api_base: str | None = None
    openai_compat_model: str | None = None
    openai_compat_api_key: str | None = None


class MemoryExtractorConfig(BaseModel):
    """TPM 记忆抽取器配置。

    该配置只控制"记忆抽取"这条链路，和主对话模型分离。
    如果启用，它会优先作为 TPM 的 LLM 抽取后端使用。
    """

    enabled: bool = False
    api_base: str | None = None
    model: str | None = None
    api_key: str | None = None
    timeout: float = 30.0
    max_candidates: int = 8


class TPMSettings(BaseModel):
    """TPM 引擎可调参数（映射到 tpm.memory.TPMConfig）。

    decay_lambdas 为空 dict 时，build_tpm_config 回退到 TPMConfig 默认值。
    """

    write_threshold: float = 0.68
    context_threshold: float = 0.62
    promote_threshold: float = 0.72
    promotion_min_sessions: int = 2
    conflict_context_threshold: float = 0.62
    conflict_value_threshold: float = 0.35
    T_fresh: float = 168.0
    history_window: int = 3
    write_weights: list[float] = [0.25, 0.3, 0.25, 0.2]
    promote_weights: list[float] = [0.35, 0.2, 0.15, 0.25, 0.2]
    retrieve_weights: list[float] = [0.35, 0.2, 0.15, 0.2, 0.1]
    decay_lambdas: dict[str, float] = Field(default_factory=dict)
    positive_reinforcement: float = 0.08
    negative_penalty: float = 0.12
    working_decay: float = 0.015
    short_term_decay: float = 0.03


def build_tpm_config(settings: TPMSettings) -> "TPMConfig":
    """把 pydantic TPMSettings 转成 tpm.memory.TPMConfig（list→tuple）。"""
    from .tpm.memory import TPMConfig

    defaults = TPMConfig()
    return TPMConfig(
        write_threshold=settings.write_threshold,
        context_threshold=settings.context_threshold,
        promote_threshold=settings.promote_threshold,
        promotion_min_sessions=settings.promotion_min_sessions,
        conflict_context_threshold=settings.conflict_context_threshold,
        conflict_value_threshold=settings.conflict_value_threshold,
        T_fresh=settings.T_fresh,
        history_window=settings.history_window,
        write_weights=tuple(settings.write_weights),
        promote_weights=tuple(settings.promote_weights),
        retrieve_weights=tuple(settings.retrieve_weights),
        decay_lambdas=dict(settings.decay_lambdas) if settings.decay_lambdas else defaults.decay_lambdas,
        positive_reinforcement=settings.positive_reinforcement,
        negative_penalty=settings.negative_penalty,
        working_decay=settings.working_decay,
        short_term_decay=settings.short_term_decay,
    )


class AgentConfig(BaseModel):
    """Agent configuration"""

    max_steps: int = 50
    workspace_dir: str = "./workspace"
    system_prompt_path: str = "system_prompt.md"


class MCPConfig(BaseModel):
    """MCP (Model Context Protocol) timeout configuration"""

    connect_timeout: float = 10.0  # Connection timeout (seconds)
    execute_timeout: float = 60.0  # Tool execution timeout (seconds)
    sse_read_timeout: float = 120.0  # SSE read timeout (seconds)


class ToolsConfig(BaseModel):
    """Tools configuration"""

    # Basic tools (file operations, bash)
    enable_file_tools: bool = True
    enable_bash: bool = True
    enable_note: bool = True

    # Skills
    enable_skills: bool = True
    skills_dir: str = "./skills"

    # MCP tools
    enable_mcp: bool = True
    mcp_config_path: str = "mcp.json"
    mcp: MCPConfig = Field(default_factory=MCPConfig)


class Config(BaseModel):
    """Main configuration class"""

    llm: LLMConfig
    agent: AgentConfig
    tools: ToolsConfig
    memory_extractor: MemoryExtractorConfig = Field(default_factory=MemoryExtractorConfig)
    tpm: TPMSettings = Field(default_factory=TPMSettings)

    @classmethod
    def load(cls) -> "Config":
        """Load configuration from the default search path."""
        config_path = cls.get_default_config_path()
        if not config_path.exists():
            raise FileNotFoundError("Configuration file not found. Run scripts/setup-config.sh or place config.yaml in mini_agent/config/.")
        return cls.from_yaml(config_path)

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> "Config":
        """Load configuration from YAML file

        Args:
            config_path: Configuration file path

        Returns:
            Config instance

        Raises:
            FileNotFoundError: Configuration file does not exist
            ValueError: Invalid configuration format or missing required fields
        """
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file does not exist: {config_path}")

        with open(config_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not data:
            raise ValueError("Configuration file is empty")

        # Parse LLM configuration
        if "api_key" not in data:
            raise ValueError("Configuration file missing required field: api_key")

        if not data["api_key"] or data["api_key"] == "YOUR_API_KEY_HERE":
            raise ValueError("Please configure a valid API Key")

        # Parse retry configuration
        retry_data = data.get("retry", {})
        retry_config = RetryConfig(
            enabled=retry_data.get("enabled", True),
            max_retries=retry_data.get("max_retries", 3),
            initial_delay=retry_data.get("initial_delay", 1.0),
            max_delay=retry_data.get("max_delay", 60.0),
            exponential_base=retry_data.get("exponential_base", 2.0),
        )

        llm_config = LLMConfig(
            api_key=data["api_key"],
            api_base=data.get("api_base", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
            model=data.get("model", "qwen2.5-7b-instruct"),
            provider=data.get("provider", "openai"),
            lora_adapter_dir=data.get("lora_adapter_dir"),
            lora_user_id=data.get("lora_user_id"),
            retry=retry_config,
            openai_compat_enabled=(data.get("openai_compat") or {}).get("enabled", False) if isinstance(data.get("openai_compat"), dict) else bool(data.get("openai_compat", False)),
            openai_compat_api_base=(data.get("openai_compat") or {}).get("api_base") if isinstance(data.get("openai_compat"), dict) else None,
            openai_compat_model=(data.get("openai_compat") or {}).get("model") if isinstance(data.get("openai_compat"), dict) else None,
            openai_compat_api_key=(data.get("openai_compat") or {}).get("api_key") if isinstance(data.get("openai_compat"), dict) else None,
        )

        memory_extractor_data = data.get("memory_extractor") or {}
        if not isinstance(memory_extractor_data, dict):
            raise ValueError("memory_extractor must be a mapping if provided")

        memory_extractor_config = MemoryExtractorConfig(
            enabled=memory_extractor_data.get("enabled", False),
            api_base=memory_extractor_data.get("api_base"),
            model=memory_extractor_data.get("model"),
            api_key=memory_extractor_data.get("api_key"),
            timeout=memory_extractor_data.get("timeout", 30.0),
            max_candidates=memory_extractor_data.get("max_candidates", 8),
        )

        tpm_data = data.get("tpm") or {}
        if not isinstance(tpm_data, dict):
            raise ValueError("tpm must be a mapping if provided")
        tpm_settings = TPMSettings(**tpm_data)

        # Parse Agent configuration
        agent_config = AgentConfig(
            max_steps=data.get("max_steps", 50),
            workspace_dir=data.get("workspace_dir", "./workspace"),
            system_prompt_path=data.get("system_prompt_path", "system_prompt.md"),
        )

        # Parse tools configuration
        tools_data = data.get("tools", {})

        # Parse MCP configuration
        mcp_data = tools_data.get("mcp", {})
        mcp_config = MCPConfig(
            connect_timeout=mcp_data.get("connect_timeout", 10.0),
            execute_timeout=mcp_data.get("execute_timeout", 60.0),
            sse_read_timeout=mcp_data.get("sse_read_timeout", 120.0),
        )

        tools_config = ToolsConfig(
            enable_file_tools=tools_data.get("enable_file_tools", True),
            enable_bash=tools_data.get("enable_bash", True),
            enable_note=tools_data.get("enable_note", True),
            enable_skills=tools_data.get("enable_skills", True),
            skills_dir=tools_data.get("skills_dir", "./skills"),
            enable_mcp=tools_data.get("enable_mcp", True),
            mcp_config_path=tools_data.get("mcp_config_path", "mcp.json"),
            mcp=mcp_config,
        )

        return cls(
            llm=llm_config,
            agent=agent_config,
            tools=tools_config,
            memory_extractor=memory_extractor_config,
            tpm=tpm_settings,
        )

    @staticmethod
    def get_package_dir() -> Path:
        """Get the package installation directory

        Returns:
            Path to the mini_agent package directory
        """
        # Get the directory where this config.py file is located
        return Path(__file__).parent

    @classmethod
    def find_config_file(cls, filename: str) -> Path | None:
        """Find configuration file with priority order

        Search for config file in the following order of priority:
        1) mini_agent/config/{filename} in current directory (development mode)
        2) ~/.mini-agent/config/{filename} in user home directory
        3) {package}/mini_agent/config/{filename} in package installation directory

        Args:
            filename: Configuration file name (e.g., "config.yaml", "mcp.json", "system_prompt.md")

        Returns:
            Path to found config file, or None if not found
        """
        config_dirs = [
            Path.cwd() / "mini_agent" / "config",
            Path.home() / ".mini-agent" / "config",
            cls.get_package_dir() / "config",
        ]

        for config_dir in config_dirs:
            config_file = config_dir / filename
            if config_file.exists():
                return config_file

        # Allow the CLI to fall back to the checked-in template when the
        # user-specific MCP config has not been created yet.
        if filename == "mcp.json":
            for config_dir in config_dirs:
                example_file = config_dir / "mcp-example.json"
                if example_file.exists():
                    return example_file

        return None

    @classmethod
    def get_default_config_path(cls) -> Path:
        """Get the default config file path with priority search

        Returns:
            Path to config.yaml (prioritizes: dev config/ > user config/ > package config/)
        """
        config_path = cls.find_config_file("config.yaml")
        if config_path:
            return config_path

        # Fallback to package config directory for error message purposes
        return cls.get_package_dir() / "config" / "config.yaml"

# TPPM 代码对齐实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 `Mini-Agent-5-1/mini_agent/tpm/` 模块按论文《时间心理画像记忆 TPPM》主文方法重构，使其成为论文方法的忠实实现（双字段心理画像、类型条件衰减+风险安全规则、Fresh+confidence 检索、可配冲突阈值、心理导向历史感知抽取、配置外置）。

**Architecture:** TPPM 保持规则驱动（无可训练参数）。`ProfileCandidate`/`ProfileMemoryUnit` 的 `profile_type` 拆为 `slot`($a_i$, 7 类心理子空间) + `memory_type`($g_i$, 5 类时间–安全类型)；衰减率按 $g_i$ 索引并新增 `is_risk` 安全旁路；检索按主文式(16) 改为 Rel+stability+Ctx+Fresh+confidence；冲突阈值可配；抽取器输出双字段并接收历史窗口；`TPMConfig` 外置到 `config.yaml`。LoRA 链路已放弃、不触碰；现有数据文件不修改，`from_dict` 仅只读向后兼容。

**Tech Stack:** Python 3.10+ dataclasses(slots), pydantic v2, PyYAML, requests, pytest, pytest-asyncio。

**关键约束（来自用户，贯穿全部任务）:**
1. **数据暂不动** — 不修改任何现有数据文件（`Table*/*_memory_bank.json`、`workspace/.agent_memory.json`、`Figure-data/` 等）。`from_dict` 仅只读向后兼容，不回写源文件。不提供批量迁移脚本。测试一律用临时 workspace。
2. **论文不改** — `draft/TPPM-draft.tex` 不做任何修改。
3. **基座 = DeepSeek-V4-Flash**（与论文附录一致）；不统一/修改 config 中的模型字符串。
4. **LoRA 已放弃、不在范围** — 不碰 `LoRA/`、`mini_agent/llm/local_lora_client.py`、`agent.py` 中 LoRA 逻辑（`_check_and_load_adapter`、`enable_background_distillation` 触发的蒸馏子进程）。`agent.py` 只允许改 `add_user_message`（传 recent history）。
5. **TPPM 规则驱动**；保持三层记忆主干、情境分支结构、证据集合、`SceneProfileBranch` 结构不变。

**参考 spec:** `docs/superpowers/specs/2026-07-15-tppm-code-alignment-design.md`
**目标代码根目录:** `/root/autodl-tmp/wangqihao/Mini-Agent-5-1/`（所有改动仅限此目录下）

**实施顺序说明:** spec §13 的顺序中「风险规则」依赖 `is_risk`（即 `slot` 字段），因此本计划把「双字段数据模型」提前到 Task 2，使后续机制任务都有 `slot`/`memory_type` 可用。最终覆盖范围与 spec 完全一致。

---

## File Structure

| 文件 | 职责 | 本计划改动 |
|---|---|---|
| `mini_agent/tpm/models.py` | PMU/候选/证据/分支数据模型 | 双字段拆分、`is_risk`、因子重命名、迁移助手、`from_dict` 兼容 |
| `mini_agent/tpm/memory.py` | 三层记忆引擎 + 管理器 + `TPMConfig` | 衰减+风险、检索 Fresh/confidence、冲突阈值、固化重命名、历史入口、配置外置接线 |
| `mini_agent/tpm/extractor.py` | 正则/LLM 候选抽取 | 心理导向 schema、中文正则、slot+type 输出、历史入参 |
| `mini_agent/agent.py` | Agent 轮次生命周期 | 仅 `add_user_message` 传 recent history |
| `mini_agent/cli.py` | 工具装配 | 读取 tpm 配置、传 `TPMConfig`、API key 环境变量回退 |
| `mini_agent/config.py` | YAML → 配置对象 | `TPMSettings` + `build_tpm_config` |
| `mini_agent/config/config.yaml` | 配置文件 | 新增 `tpm:` 块、移除明文 key |
| `tests/test_tpm_memory.py` | TPM 测试 | 更新既有用例字段名 + 新增机制用例 |

**回归检查点:** 每个 Task 末尾运行 `cd /root/autodl-tmp/wangqihao/Mini-Agent-5-1 && python -m pytest tests/test_tpm_memory.py -v`，必须全绿才提交。

---

## Task 1: 配置外置（可调参通路，纯增量）

**Files:**
- Modify: `mini_agent/tpm/memory.py`（`TPMConfig` 加字段 + `TemporalProfileMemory.to_dict`/`from_dict` 序列化新字段 + `TPMMemoryManager` 暴露 `history_window`）
- Modify: `mini_agent/config.py`（新增 `TPMSettings` + `build_tpm_config` + `Config.from_yaml` 解析 `tpm:` 块）
- Modify: `mini_agent/config/config.yaml`（新增 `tpm:` 块，暂不含 `decay_lambdas`）
- Modify: `mini_agent/cli.py`（构造 `TPMConfig` 并传入 `TPMMemoryManager`）
- Test: `tests/test_tpm_memory.py`

- [ ] **Step 1: 写失败测试 — 配置外置与新字段**

在 `tests/test_tpm_memory.py` 末尾追加：

```python
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
```

- [ ] **Step 2: 运行测试，确认失败**

Run: `cd /root/autodl-tmp/wangqihao/Mini-Agent-5-1 && python -m pytest tests/test_tpm_memory.py::test_tpm_config_externalized_from_yaml tests/test_tpm_memory.py::test_tpm_manager_exposes_history_window_from_config tests/test_tpm_memory.py::test_tpm_config_roundtrips_new_fields_through_to_dict -v`
Expected: FAIL（`TPMConfig` 无 `T_fresh`/`history_window`/`conflict_*` 字段；`Config` 无 `tpm`；`build_tpm_config` 未定义）

- [ ] **Step 3: 给 `TPMConfig` 增加新字段**

在 `mini_agent/tpm/memory.py` 的 `TPMConfig` 末尾（`short_term_decay` 之后）追加 4 个字段：

```python
    working_decay: float = 0.015
    short_term_decay: float = 0.03
    # --- 论文主文对齐新增（Task 1）---
    conflict_context_threshold: float = 0.62   # δ_ctx：情境重叠阈值
    conflict_value_threshold: float = 0.35     # 极性分歧阈值
    T_fresh: float = 168.0                      # Fresh 衰减时间常数（小时）
    history_window: int = 3                     # 历史感知窗口 N
```

- [ ] **Step 4: 在 `TemporalProfileMemory.to_dict` 的 config 块追加新字段**

在 `mini_agent/tpm/memory.py` 的 `to_dict` 方法 `"config"` 字典里，`"short_term_decay"` 行之后追加：

```python
                "short_term_decay": self.config.short_term_decay,
                "conflict_context_threshold": self.config.conflict_context_threshold,
                "conflict_value_threshold": self.config.conflict_value_threshold,
                "T_fresh": self.config.T_fresh,
                "history_window": self.config.history_window,
```

- [ ] **Step 5: 在 `TemporalProfileMemory.from_dict` 解析新字段**

在 `from_dict` 的 `TPMConfig(...)` 构造里，`short_term_decay=...` 行之后追加：

```python
            short_term_decay=config_data.get("short_term_decay", default_config.short_term_decay),
            conflict_context_threshold=config_data.get(
                "conflict_context_threshold", default_config.conflict_context_threshold
            ),
            conflict_value_threshold=config_data.get(
                "conflict_value_threshold", default_config.conflict_value_threshold
            ),
            T_fresh=config_data.get("T_fresh", default_config.T_fresh),
            history_window=config_data.get("history_window", default_config.history_window),
        )
```

- [ ] **Step 6: `TPMMemoryManager` 暴露 `history_window`**

在 `mini_agent/tpm/memory.py` 的 `TPMMemoryManager.__init__` 中，`self.memory = TemporalProfileMemory(config=config)` 之后追加一行：

```python
        self.memory = TemporalProfileMemory(config=config)
        self.history_window = (config or TPMConfig()).history_window
```

- [ ] **Step 7: 在 `config.py` 增加 `TPMSettings` 与 `build_tpm_config`**

在 `mini_agent/config.py` 顶部 import 区之后（`MemoryExtractorConfig` 类定义之后）新增：

```python
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
```

- [ ] **Step 8: 在 `Config` 类挂 `tpm` 字段并解析**

在 `mini_agent/config.py` 的 `Config` 类中，把 `memory_extractor` 字段行改为同时声明 `tpm`：

```python
    llm: LLMConfig
    agent: AgentConfig
    tools: ToolsConfig
    memory_extractor: MemoryExtractorConfig = Field(default_factory=MemoryExtractorConfig)
    tpm: TPMSettings = Field(default_factory=TPMSettings)
```

在 `from_yaml` 的 `return cls(...)` 处，增加 `tpm` 构造与传参。先在 `memory_extractor_config = ...` 之后插入解析：

```python
        tpm_data = data.get("tpm") or {}
        if not isinstance(tpm_data, dict):
            raise ValueError("tpm must be a mapping if provided")
        tpm_settings = TPMSettings(**tpm_data)
```

再把 `return cls(...)` 改为：

```python
        return cls(
            llm=llm_config,
            agent=agent_config,
            tools=tools_config,
            memory_extractor=memory_extractor_config,
            tpm=tpm_settings,
        )
```

- [ ] **Step 9: 在 `config.yaml` 新增 `tpm:` 块**

在 `mini_agent/config/config.yaml` 的 `memory_extractor:` 块之后（`# Third-party provider examples` 注释之前）插入：

```yaml
# ===== TPPM 引擎参数（论文主文对齐）=====
# 省略的字段使用 tpm/memory.py 中 TPMConfig 的默认值。
tpm:
  write_threshold: 0.68        # 论文 θ_write
  context_threshold: 0.62      # 对齐匹配阈值（附录）
  promote_threshold: 0.72      # 论文 θ_promote
  promotion_min_sessions: 2    # 论文 K_sess
  conflict_context_threshold: 0.62  # δ_ctx：情境重叠阈值
  conflict_value_threshold: 0.35     # 极性分歧阈值
  T_fresh: 168.0               # Fresh 时间常数（小时）
  history_window: 3            # 历史感知窗口 N
  write_weights: [0.25, 0.3, 0.25, 0.2]       # relevance, explicitness, utility, stability
  promote_weights: [0.35, 0.2, 0.15, 0.25, 0.2]  # reinforcement, explicitness, utility, stability, contradiction
  retrieve_weights: [0.35, 0.2, 0.15, 0.2, 0.1]  # rel, stability, ctx, fresh, confidence
  positive_reinforcement: 0.08
  negative_penalty: 0.12
  working_decay: 0.015
  short_term_decay: 0.03
```

- [ ] **Step 10: `cli.py` 构造 `TPMConfig` 并传入 manager**

在 `mini_agent/cli.py` 顶部确认有 `import os`（若无则添加到现有 import 区）。再在 `add_workspace_tools` 中，把构造 `TPMMemoryManager` 的代码（当前约 537 行）改为传入 config：

```python
        tpm_config = build_tpm_config(config.tpm)
        memory_manager = TPMMemoryManager(
            memory_file=str(workspace_dir / ".agent_memory.json"),
            extractor=extractor,
            config=tpm_config,
        )
```

并在 `cli.py` 顶部 import 区加上 `from .config import build_tpm_config`（若 cli 已 `from .config import Config`，则改为 `from .config import Config, build_tpm_config`）。

- [ ] **Step 11: 运行测试，确认通过**

Run: `cd /root/autodl-tmp/wangqihao/Mini-Agent-5-1 && python -m pytest tests/test_tpm_memory.py -v`
Expected: 全部 PASS（含 3 个新测试 + 既有测试）。

- [ ] **Step 12: 提交**

```bash
cd /root/autodl-tmp/wangqihao
git add Mini-Agent-5-1/mini_agent/tpm/memory.py Mini-Agent-5-1/mini_agent/config.py Mini-Agent-5-1/mini_agent/config/config.yaml Mini-Agent-5-1/mini_agent/cli.py Mini-Agent-5-1/tests/test_tpm_memory.py
git commit -m "feat(tpm): externalize TPMConfig to config.yaml (additive: conflict_*, T_fresh, history_window)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 2: 双字段数据模型 + 因子重命名（基础结构）

把 `profile_type` 拆为 `slot`+`memory_type`；`recency`→`relevance`、`user_relevance`→`utility`；新增 `is_risk`；`from_dict` 只读迁移旧 schema；更新所有内部引用与既有测试。这是后续机制任务的基础。

**Files:**
- Modify: `mini_agent/tpm/models.py`（核心）
- Modify: `mini_agent/tpm/memory.py`（所有 `profile_type`/`recency`/`user_relevance` 调用点）
- Modify: `mini_agent/tpm/extractor.py`（`_parse_candidates` 读新字段 + 迁移；`RegexProfileExtractor` 规格转双字段；`_default_stability` 按 slot）
- Test: `tests/test_tpm_memory.py`（更新 #2/#3；新增迁移/回填测试）

- [ ] **Step 1: 写失败测试 — 双字段迁移与回填**

在 `tests/test_tpm_memory.py` 末尾追加：

```python
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
```

- [ ] **Step 2: 运行测试，确认失败**

Run: `cd /root/autodl-tmp/wangqihao/Mini-Agent-5-1 && python -m pytest tests/test_tpm_memory.py -v`
Expected: FAIL（`ProfileCandidate`/`ProfileMemoryUnit` 无 `slot`/`memory_type`/`is_risk`/`relevance`/`utility`；`default_memory_type` 未定义）

- [ ] **Step 3: 在 `models.py` 顶部增加常量与迁移助手**

在 `mini_agent/tpm/models.py` 的 `utc_now()` 函数之后插入：

```python
# 论文表1：7 个心理子空间 a_i
SLOT_VALUES = {"affect", "stressor", "cognitive", "coping", "support", "behavior", "risk"}
# 论文式(15)：5 个时间–安全类型 g_i
MEMORY_TYPE_VALUES = {"affect", "stressor", "coping", "support", "trait"}

# §4.1 默认 slot -> memory_type 映射（抽取器只给 slot 时回填 g_i）
DEFAULT_MEMORY_TYPE = {
    "affect": "affect",
    "stressor": "stressor",
    "coping": "coping",
    "support": "support",
    "cognitive": "trait",
    "behavior": "coping",
    "risk": "affect",
}

# §4.2 旧 profile_type -> {slot, memory_type} 迁移表（仅 from_dict 内存解析，不回写）
LEGACY_PROFILE_TYPE_MAP = {
    "background": ("support", "trait"),
    "preference": ("cognitive", "trait"),
    "goal": ("behavior", "coping"),
    "style": ("cognitive", "trait"),
    "interest": ("behavior", "trait"),
    "general": ("coping", "trait"),
}


def default_memory_type(slot: str) -> str:
    """按 §4.1 回填 memory_type；未知 slot 默认 trait。"""
    return DEFAULT_MEMORY_TYPE.get(slot, "trait")


def migrate_profile_type(legacy: str) -> tuple[str, str]:
    """把旧 profile_type（或裸 slot/g_i）解析为 (slot, memory_type)。best-effort。"""
    legacy = (legacy or "general").strip().lower()
    if legacy in LEGACY_PROFILE_TYPE_MAP:
        return LEGACY_PROFILE_TYPE_MAP[legacy]
    if legacy in SLOT_VALUES:
        return (legacy, default_memory_type(legacy))
    if legacy in MEMORY_TYPE_VALUES:
        return ("coping", legacy)
    return ("coping", "trait")
```

- [ ] **Step 4: 重写 `ProfileCandidate`（拆双字段 + 重命名因子）**

把 `mini_agent/tpm/models.py` 中整个 `ProfileCandidate` 类替换为：

```python
@dataclass(slots=True)
class ProfileCandidate:
    """Candidate profile item extracted from conversation."""

    attribute: str
    value: str
    context: str
    slot: str
    memory_type: str
    scene: str = "general"
    confidence: float = 0.7
    stability: float = 0.5
    relevance: float = 1.0
    explicitness: float = 0.7
    utility: float = 0.75
    source: str = "user_utterance"
    timestamp: str = field(default_factory=lambda: utc_now().isoformat())

    def write_score(self, weights: tuple[float, float, float, float]) -> float:
        """Compute the write score (论文式8: φ = α1·r + α2·e + α3·u + α4·b)."""
        alpha1, alpha2, alpha3, alpha4 = weights
        return (
            alpha1 * self.relevance
            + alpha2 * self.explicitness
            + alpha3 * self.utility
            + alpha4 * self.stability
        )

    @property
    def quality_score(self) -> float:
        """Quality proxy used by explicit TPM rules."""
        return (self.confidence + self.explicitness + self.utility) / 3.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProfileCandidate":
        payload = dict(data)
        if "slot" not in payload:
            legacy = payload.pop("profile_type", "general")
            payload["slot"], payload["memory_type"] = migrate_profile_type(legacy)
        elif "memory_type" not in payload:
            payload["memory_type"] = default_memory_type(payload["slot"])
        payload.pop("profile_type", None)
        # 重命名因子向后兼容
        if "relevance" not in payload and "recency" in payload:
            payload["relevance"] = payload.pop("recency")
        if "utility" not in payload and "user_relevance" in payload:
            payload["utility"] = payload.pop("user_relevance")
        payload.pop("recency", None)
        payload.pop("user_relevance", None)
        return cls(**payload)
```

- [ ] **Step 5: 重写 `ProfileMemoryUnit` 字段与 `is_risk`、`to_dict`、`from_dict`**

在 `mini_agent/tpm/models.py` 的 `ProfileMemoryUnit` 类中：
(a) 把字段 `profile_type: str` 替换为两行：

```python
    slot: str
    memory_type: str
```

(b) 在 `memory_level: str = "working"` 字段之后，新增 `is_risk` 派生属性（放在 `ensure_branch` 方法之前）：

```python
    @property
    def is_risk(self) -> bool:
        """论文风险安全规则：slot 属于风险信号子空间时为真。"""
        return self.slot == "risk"
```

(c) 在 `to_dict` 中，把 `"profile_type": self.profile_type,` 行替换为：

```python
            "slot": self.slot,
            "memory_type": self.memory_type,
```

(d) 在 `from_dict` 中，把构造 `cls(...)` 里的 `profile_type=payload.get("profile_type", "general"),` 行替换为双字段解析。在 `payload = dict(data)` 之后、构造 `cls(...)` 之前插入：

```python
        if "slot" not in payload:
            legacy = payload.pop("profile_type", "general")
            payload["slot"], payload["memory_type"] = migrate_profile_type(legacy)
        elif "memory_type" not in payload:
            payload["memory_type"] = default_memory_type(payload["slot"])
        payload.pop("profile_type", None)
```

并把 `cls(...)` 调用中的 `profile_type=payload.get("profile_type", "general"),` 改为：

```python
            slot=payload["slot"],
            memory_type=payload["memory_type"],
```

- [ ] **Step 6: 更新 `memory.py` 所有 `profile_type`/`recency`/`user_relevance` 调用点**

在 `mini_agent/tpm/memory.py` 顶部 import 行，把
`from .models import EvidenceItem, ProfileCandidate, ProfileMemoryUnit, utc_now`
改为：

```python
from .models import (
    EvidenceItem,
    ProfileCandidate,
    ProfileMemoryUnit,
    default_memory_type,
    migrate_profile_type,
    utc_now,
)
```

然后逐处修改（按文件中顺序）：

1. `decay_long_term` 中 `decay = self.config.decay_lambdas.get(unit.profile_type, self.config.decay_lambdas["general"])` 改为：

```python
            decay = self.config.decay_lambdas.get(unit.memory_type, self.config.decay_lambdas.get("trait", 0.03))
```

（风险旁路在 Task 3 实现；此处仅改索引键，避免 KeyError。）

2. `distillation_payload` 中 `"profile_type": unit.profile_type,` 改为：

```python
                    "slot": unit.slot,
                    "memory_type": unit.memory_type,
```

3. `_align_or_create` 中构造 `ProfileMemoryUnit(...)` 的 `profile_type=candidate.profile_type,` 改为：

```python
                slot=candidate.slot,
                memory_type=candidate.memory_type,
```

4. `_similarity` 中 `type_score = 1.0 if candidate.profile_type == unit.profile_type else 0.35` 改为：

```python
        type_score = 1.0 if candidate.slot == unit.slot else 0.35
```

5. `_merge_into_store` 中 `and existing.profile_type == unit.profile_type` 改为 `and existing.slot == unit.slot`。

6. `_retrieve_score` 中 `1.0 if unit.attribute in query_norm or unit.profile_type in query_norm else 0.0,` 这一行删除（Fresh+confidence 改写在 Task 4；此处先把 `profile_type` 引用移除，临时把 `ctx_score` 简化，避免引用已删除字段）。把 `ctx_score = max(...)` 整块替换为：

```python
        ctx_score = max(
            _similarity(query_norm, branch.context),
            _similarity(query_norm, unit.context),
        )
```

7. `augment_user_message` 中 `f"(type={item.profile_type}, scene={branch.scene}, "` 改为 `f"(slot={item.slot}, type={item.memory_type}, scene={branch.scene}, "`。

8. `recall` 中 `or category_norm in item.profile_type.lower()` 改为 `or category_norm in item.slot.lower() or category_norm in item.memory_type.lower()`。

9. `format_recall` 中 `f"   (type={item.profile_type}, level={item.memory_level}, "` 改为 `f"   (slot={item.slot}, type={item.memory_type}, level={item.memory_level}, "`。

10. `_fallback_candidate` 整体替换为（category→双字段映射 + 重命名因子）：

```python
    def _fallback_candidate(self, content: str, category: str) -> ProfileCandidate:
        category_norm = (category or "general").lower()
        category_map = {
            "user_preference": ("cognitive", "trait"),
            "preference": ("cognitive", "trait"),
            "project_info": ("behavior", "coping"),
            "decision": ("behavior", "coping"),
            "background": ("support", "trait"),
            "identity": ("support", "trait"),
            "style": ("cognitive", "trait"),
        }
        slot, memory_type = category_map.get(category_norm, ("coping", "trait"))
        attribute = category_norm if category_norm != "general" else "explicit_note"
        return ProfileCandidate(
            attribute=attribute,
            value=content,
            context=content,
            slot=slot,
            memory_type=memory_type,
            scene=category or "general",
            confidence=0.9,
            stability=0.72,
            explicitness=0.96,
            relevance=1.0,
            utility=0.95,
            source="manual_note",
        )
```

- [ ] **Step 7: 更新 `extractor.py` 的 `_parse_candidates` 读新字段 + 迁移**

在 `mini_agent/tpm/extractor.py` 顶部 import 行，把
`from .models import ProfileCandidate`
改为：

```python
from .models import (
    MEMORY_TYPE_VALUES,
    SLOT_VALUES,
    ProfileCandidate,
    default_memory_type,
    migrate_profile_type,
)
```

把 `_parse_candidates` 方法整体替换为：

```python
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
```

并把 `_default_stability` 方法替换为按 slot：

```python
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
```

- [ ] **Step 8: 更新 `RegexProfileExtractor` 规格为双字段**

把 `mini_agent/tpm/extractor.py` 的 `RegexProfileExtractor.extract` 方法整体替换为（支持带/不带捕获组；英文规则转双字段）：

```python
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
```

- [ ] **Step 9: 更新既有测试 #2（LLM 抽取器结构化用例）**

在 `tests/test_tpm_memory.py` 的 `test_llm_profile_extractor_parses_structured_candidates` 中，把 payload 的 candidate 字段与断言改为新 schema。把 `"profile_type": "style",` 行替换为：

```python
                          "slot": "cognitive",
                          "memory_type": "trait",
```

把 `"recency": 1.0,` 改为 `"relevance": 1.0,`；把 `"user_relevance": 0.94,` 改为 `"utility": 0.94,`。把末尾断言 `assert candidates[0].profile_type == "style"` 替换为：

```python
    assert candidates[0].slot == "cognitive"
    assert candidates[0].memory_type == "trait"
```

- [ ] **Step 10: 更新既有测试 #3（场景分支/会话计数用例）**

在 `tests/test_tpm_memory.py` 的 `test_tpm_scene_branches_and_session_count_are_explicit` 中，把两处 `ProfileCandidate(...)` 构造里的 `profile_type="style",` 替换为 `slot="cognitive", memory_type="trait",`，并把 `user_relevance=0.93`/`0.9`/`0.92` 分别替换为 `utility=0.93`/`0.9`/`0.92`（共三处构造）。

- [ ] **Step 11: 运行全量测试，确认通过**

Run: `cd /root/autodl-tmp/wangqihao/Mini-Agent-5-1 && python -m pytest tests/test_tpm_memory.py -v`
Expected: 全部 PASS。

- [ ] **Step 12: 提交**

```bash
cd /root/autodl-tmp/wangqihao
git add Mini-Agent-5-1/mini_agent/tpm/models.py Mini-Agent-5-1/mini_agent/tpm/memory.py Mini-Agent-5-1/mini_agent/tpm/extractor.py Mini-Agent-5-1/tests/test_tpm_memory.py
git commit -m "refactor(tpm): split profile_type into slot+memory_type, rename write/promote factors

- ProfileCandidate/ProfileMemoryUnit: slot(a_i, 7 psych subspaces) + memory_type(g_i, 5 time-safety types)
- add is_risk property; from_dict read-only legacy migration (no source file rewrite)
- rename recency->relevance, user_relevance->utility (align paper main text r/u)
- decay_lambdas indexed by memory_type; safe default fallback
- extractor reads slot/memory_type with legacy fallback; regex specs dual-field

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 3: 风险安全规则 + 类型条件衰减（论文式14、15）

`is_risk` 单元跳过常规时间衰减；仅在命中 contradiction 时由 $-\gamma^-\psi^-$ 降强度（R1：复用 contradiction 路径）。

**Files:**
- Modify: `mini_agent/tpm/memory.py`（`decay_long_term`）
- Test: `tests/test_tpm_memory.py`

- [ ] **Step 1: 写失败测试 — 风险单元不随时间衰减，仅 contradiction 降强度**

在 `tests/test_tpm_memory.py` 末尾追加：

```python
def test_risk_unit_skips_time_decay_but_drops_on_contradiction():
    from datetime import timedelta
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
    from datetime import timedelta
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
```

在 `tests/test_tpm_memory.py` 顶部 import 区追加（若尚未导入）：

```python
from datetime import timedelta
from mini_agent.tpm.models import utc_now
```

（若 `utc_now` 与已有 import 冲突，合并到现有 `from mini_agent.tpm.models import ...` 行。）

- [ ] **Step 2: 运行测试，确认失败**

Run: `cd /root/autodl-tmp/wangqihao/Mini-Agent-5-1 && python -m pytest tests/test_tpm_memory.py::test_risk_unit_skips_time_decay_but_drops_on_contradiction tests/test_tpm_memory.py::test_non_risk_affect_unit_decays_over_time -v`
Expected: 风险测试 FAIL（当前 `decay_long_term` 对 risk 单元也做 exp 衰减 → 强度下降，断言 `==0.8` 失败）；`test_decay_lambdas_default_keys_are_g_i_per_paper_eq15` 也 FAIL（当前默认键仍是 legacy profile_type 键，不等于 g_i 集合）。

- [ ] **Step 3: 实现风险旁路 + 把 `decay_lambdas` 默认键改为 g_i（论文式15）**

**(a) 先把 `TPMConfig.decay_lambdas` 默认值从旧 profile_type 键改为 g_i 键**（spec §5.1；当前仍是 legacy `{goal, interest, style, background, preference, general}`，会导致 `decay_long_term` 中 `decay_lambdas.get(unit.memory_type, ...)` 全部回退到 0.03，类型条件衰减失效）。在 `mini_agent/tpm/memory.py` 把 `decay_lambdas` 字段整体替换为：

```python
    decay_lambdas: dict[str, float] = field(
        default_factory=lambda: {
            "affect": 0.10,
            "stressor": 0.07,
            "coping": 0.05,
            "support": 0.05,
            "trait": 0.03,
        }
    )
```

满足论文式(15) 排序：$\lambda_{\text{affect}}>\lambda_{\text{stressor}}>\lambda_{\text{coping}}\approx\lambda_{\text{support}}>\lambda_{\text{trait}}$。

**(b) 再把 `decay_long_term` 方法整体替换为**：

```python
    def decay_long_term(self) -> None:
        now = utc_now()
        for unit in self.long_term_memory:
            delta_hours = max((now - _parse_timestamp(unit.last_evolved)).total_seconds() / 3600.0, 0.0)
            if delta_hours <= 0:
                continue

            if unit.is_risk:
                # 论文风险安全规则：风险信号子空间关闭常规时间衰减；
                # 仅当命中 contradiction(ψ⁻) 时由 -γ⁻·ψ⁻ 降低强度（R1 反证）。
                negative_signal = min(1.0, unit.contradiction_count / max(1, unit.reinforcement_count))
                unit.stability_score = _clamp(
                    unit.stability_score - self.config.negative_penalty * negative_signal
                )
                unit.last_evolved = now.isoformat()
                continue

            decay = self.config.decay_lambdas.get(unit.memory_type, self.config.decay_lambdas.get("trait", 0.03))
            positive_signal = min(1.0, unit.reinforcement_count / max(1, unit.session_count * 2))
            negative_signal = min(1.0, unit.contradiction_count / max(1, unit.reinforcement_count))
            unit.stability_score = _clamp(
                unit.stability_score * math.exp(-decay * delta_hours / 24.0)
                + self.config.positive_reinforcement * positive_signal
                - self.config.negative_penalty * negative_signal
            )
            unit.last_evolved = now.isoformat()
```

- [ ] **Step 4: 运行测试，确认通过**

Run: `cd /root/autodl-tmp/wangqihao/Mini-Agent-5-1 && python -m pytest tests/test_tpm_memory.py -v`
Expected: 全部 PASS。

- [ ] **Step 5: 提交**

```bash
cd /root/autodl-tmp/wangqihao
git add Mini-Agent-5-1/mini_agent/tpm/memory.py Mini-Agent-5-1/tests/test_tpm_memory.py
git commit -m "feat(tpm): risk-signal safety rule + g_i decay_lambdas (paper eq 14/15, R1)

risk units (slot=risk) bypass normal s·exp(-λΔt) decay; stability drops only
via -γ⁻·ψ⁻ when contradiction is hit. non-risk units decay by memory_type λ.
decay_lambdas default keys switched to g_i {affect,stressor,coping,support,trait}
per paper eq 15, so type-conditional decay is effective (was silently 0.03).

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 4: Fresh + confidence 检索（论文主文式16、17）

5 因子改为 `η1·Rel + η2·stability + η3·Ctx + η4·Fresh + η5·confidence`。`Rel` 并入文本相似度（value+context）；`Ctx`=场景匹配；`Fresh=exp(-Δt/T_fresh)`；第5因子由 `quality_score` 改为 `confidence_score`。

**Files:**
- Modify: `mini_agent/tpm/memory.py`（`_retrieve_score` + `retrieve_weights` 注释）
- Test: `tests/test_tpm_memory.py`

- [ ] **Step 1: 写失败测试 — Fresh 让更近访问的单元排名更高**

在 `tests/test_tpm_memory.py` 末尾追加：

```python
def test_freshness_breaks_tie_toward_recently_accessed_unit():
    from datetime import timedelta
    from mini_agent.tpm import TemporalProfileMemory
    from mini_agent.tpm.memory import TPMConfig
    from mini_agent.tpm.models import ProfileMemoryUnit

    memory = TemporalProfileMemory(config=TPMConfig())
    now = utc_now()
    old = (now - timedelta(hours=240)).isoformat()

    fresh_unit = ProfileMemoryUnit(
        attribute="mood", value="calm", context="feeling calm",
        slot="affect", memory_type="affect",
        stability_score=0.6, confidence_score=0.7, scene="general", quality_score=0.6,
        last_accessed=now.isoformat(), last_evolved=now.isoformat(),
        memory_level="long_term",
    )
    stale_unit = ProfileMemoryUnit(
        attribute="mood", value="calm", context="feeling calm",
        slot="affect", memory_type="affect",
        stability_score=0.6, confidence_score=0.7, scene="general", quality_score=0.6,
        last_accessed=old, last_evolved=old,
        memory_level="long_term",
    )
    memory.long_term_memory.extend([fresh_unit, stale_unit])

    results = memory.retrieve("calm", scene="general", top_k=2)
    assert results[0].unit_id == fresh_unit.unit_id
    assert results[1].unit_id == stale_unit.unit_id


def test_retrieve_score_uses_confidence_not_quality():
    from mini_agent.tpm import TemporalProfileMemory
    from mini_agent.tpm.memory import TPMConfig
    from mini_agent.tpm.models import ProfileMemoryUnit

    # retrieve_weights 默认第5因子=0.1；confidence 高的应排前
    memory = TemporalProfileMemory(config=TPMConfig())
    now = utc_now().isoformat()
    hi = ProfileMemoryUnit(
        attribute="mood", value="calm", context="calm",
        slot="affect", memory_type="affect",
        stability_score=0.6, confidence_score=0.95, scene="general", quality_score=0.2,
        last_accessed=now, last_evolved=now, memory_level="long_term",
    )
    lo = ProfileMemoryUnit(
        attribute="mood", value="calm", context="calm",
        slot="affect", memory_type="affect",
        stability_score=0.6, confidence_score=0.2, scene="general", quality_score=0.95,
        last_accessed=now, last_evolved=now, memory_level="long_term",
    )
    memory.long_term_memory.extend([hi, lo])
    results = memory.retrieve("calm", scene="general", top_k=2)
    assert results[0].unit_id == hi.unit_id
```

- [ ] **Step 2: 运行测试，确认失败**

Run: `cd /root/autodl-tmp/wangqihao/Mini-Agent-5-1 && python -m pytest tests/test_tpm_memory.py::test_freshness_breaks_tie_toward_recently_accessed_unit tests/test_tpm_memory.py::test_retrieve_score_uses_confidence_not_quality -v`
Expected: FAIL（当前第4因子是 scene_score、第5因子是 quality_score，无 Fresh；confidence 高者未必排前）。

- [ ] **Step 3: 重写 `_retrieve_score`**

把 `mini_agent/tpm/memory.py` 的 `_retrieve_score` 方法整体替换为：

```python
    def _retrieve_score(self, query_norm: str, unit: ProfileMemoryUnit, scene: str) -> float:
        # 论文主文式(16): Score = η1·Rel + η2·stability + η3·Ctx + η4·Fresh + η5·confidence
        branch = unit.scene_view(scene)
        rel = max(
            _similarity(query_norm, branch.value),
            _similarity(query_norm, unit.value),
            _similarity(query_norm, branch.context),
            _similarity(query_norm, unit.context),
        )
        ctx = 1.0 if branch.scene == scene else (0.7 if branch.scene == "general" or scene == "general" else 0.4)
        now = utc_now()
        delta_hours = max((now - _parse_timestamp(unit.last_accessed)).total_seconds() / 3600.0, 0.0)
        fresh = math.exp(-delta_hours / self.config.T_fresh)
        w1, w2, w3, w4, w5 = self.config.retrieve_weights
        return (
            w1 * rel
            + w2 * unit.stability_score
            + w3 * ctx
            + w4 * fresh
            + w5 * unit.confidence_score
        )
```

- [ ] **Step 4: 更新 `retrieve_weights` 注释**

在 `mini_agent/tpm/memory.py` 的 `TPMConfig` 中，把
`retrieve_weights: tuple[float, float, float, float, float] = (0.35, 0.2, 0.15, 0.2, 0.1)`
改为（仅加注释，值不变）：

```python
    # (rel, stability, ctx, fresh, confidence) — 论文主文式(16)
    retrieve_weights: tuple[float, float, float, float, float] = (0.35, 0.2, 0.15, 0.2, 0.1)
```

- [ ] **Step 5: 运行测试，确认通过**

Run: `cd /root/autodl-tmp/wangqihao/Mini-Agent-5-1 && python -m pytest tests/test_tpm_memory.py -v`
Expected: 全部 PASS。

- [ ] **Step 6: 提交**

```bash
cd /root/autodl-tmp/wangqihao
git add Mini-Agent-5-1/mini_agent/tpm/memory.py Mini-Agent-5-1/tests/test_tpm_memory.py
git commit -m "feat(tpm): Fresh + confidence retrieval (paper main-text eq 16)

5-factor score = η1·Rel + η2·stability + η3·Ctx + η4·Fresh + η5·confidence.
Rel folds value+context text sim; Ctx=scene match; Fresh=exp(-Δt/T_fresh);
5th factor switched from quality_score to confidence_score.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 5: 可配冲突阈值 + 条件分支语义（论文式10、11 + 主文 δ_ctx）

`_fuse_candidate` 冲突判定从硬编码 0.35 改为配置：情境重叠 $\rho>\delta_{ctx}$ 且值相似度 $<\delta_{val}$ 且同属性 → 冲突更新；情境不重叠 → 建立条件分支。

**Files:**
- Modify: `mini_agent/tpm/memory.py`（`_fuse_candidate`）
- Test: `tests/test_tpm_memory.py`

- [ ] **Step 1: 写失败测试 — 冲突 vs 条件分支**

在 `tests/test_tpm_memory.py` 末尾追加：

```python
def test_fuse_candidate_conflict_on_overlapping_context_divergent_value():
    from mini_agent.tpm import TemporalProfileMemory
    from mini_agent.tpm.models import EvidenceItem, ProfileCandidate

    memory = TemporalProfileMemory()
    memory.start_session("general", session_id="s1")
    seed = ProfileCandidate(
        attribute="hobby", value="我喜欢跑步", context="聊到周末运动安排",
        slot="behavior", memory_type="trait", scene="general",
        confidence=0.8, stability=0.7, explicitness=0.9, utility=0.8,
    )
    unit = memory._align_or_create(seed, session_id="s1")

    # 同情境 + 值分歧 -> 冲突
    contra = ProfileCandidate(
        attribute="hobby", value="我讨厌看书", context="聊到周末运动安排",
        slot="behavior", memory_type="trait", scene="general",
        confidence=0.85, stability=0.7, explicitness=0.9, utility=0.8,
    )
    ev1 = EvidenceItem(source="test", content=contra.context, scene="general")
    memory._fuse_candidate(unit, contra, ev1, session_id="s1")
    assert unit.contradiction_count == 1


def test_fuse_candidate_branches_when_context_does_not_overlap():
    from mini_agent.tpm import TemporalProfileMemory
    from mini_agent.tpm.models import EvidenceItem, ProfileCandidate

    memory = TemporalProfileMemory()
    memory.start_session("general", session_id="s1")
    seed = ProfileCandidate(
        attribute="hobby", value="我喜欢跑步", context="聊到周末运动安排",
        slot="behavior", memory_type="trait", scene="general",
        confidence=0.8, stability=0.7, explicitness=0.9, utility=0.8,
    )
    unit = memory._align_or_create(seed, session_id="s1")
    base_contradiction = unit.contradiction_count

    # 不同情境 + 不同 scene -> 条件分支，非冲突
    variant = ProfileCandidate(
        attribute="hobby", value="我讨厌看书", context="聊到工作日的阅读习惯",
        slot="behavior", memory_type="trait", scene="work",
        confidence=0.8, stability=0.7, explicitness=0.9, utility=0.8,
    )
    ev2 = EvidenceItem(source="test", content=variant.context, scene="work")
    memory._fuse_candidate(unit, variant, ev2, session_id="s1")
    assert unit.contradiction_count == base_contradiction
    assert "work" in unit.scene_branches


def test_fuse_candidate_uses_configurable_thresholds():
    from mini_agent.tpm import TemporalProfileMemory
    from mini_agent.tpm.memory import TPMConfig
    from mini_agent.tpm.models import EvidenceItem, ProfileCandidate

    # 抬高 value 阈值 -> 原本冲突的轻微分歧不再判冲突
    memory = TemporalProfileMemory(config=TPMConfig(conflict_value_threshold=0.05))
    memory.start_session("general", session_id="s1")
    seed = ProfileCandidate(
        attribute="hobby", value="我喜欢跑步", context="聊到周末运动安排",
        slot="behavior", memory_type="trait", scene="general",
        confidence=0.8, stability=0.7, explicitness=0.9, utility=0.8,
    )
    unit = memory._align_or_create(seed, session_id="s1")
    contra = ProfileCandidate(
        attribute="hobby", value="我讨厌看书", context="聊到周末运动安排",
        slot="behavior", memory_type="trait", scene="general",
        confidence=0.85, stability=0.7, explicitness=0.9, utility=0.8,
    )
    ev = EvidenceItem(source="test", content=contra.context, scene="general")
    memory._fuse_candidate(unit, contra, ev, session_id="s1")
    assert unit.contradiction_count == 0
```

- [ ] **Step 2: 运行测试，确认失败**

Run: `cd /root/autodl-tmp/wangqihao/Mini-Agent-5-1 && python -m pytest tests/test_tpm_memory.py::test_fuse_candidate_conflict_on_overlapping_context_divergent_value tests/test_tpm_memory.py::test_fuse_candidate_branches_when_context_does_not_overlap tests/test_tpm_memory.py::test_fuse_candidate_uses_configurable_thresholds -v`
Expected: FAIL（当前硬编码 0.35，未按 context_overlap 门控；阈值不可配）。

- [ ] **Step 3: 重写 `_fuse_candidate` 的冲突判定段**

在 `mini_agent/tpm/memory.py` 的 `_fuse_candidate` 方法中，把从 `branch = unit.ensure_branch(...)` 之后到 `if contradiction:` 之前的 `contradiction = bool(...)` 整块替换为：

```python
        branch = unit.ensure_branch(
            candidate.scene,
            value=candidate.value,
            context=candidate.context,
            confidence_score=candidate.confidence,
            quality_score=candidate.quality_score,
        )
        # 论文主文式(10/11)+δ_ctx：情境重叠 ρ>δ_ctx 且值相似度<δ_val 且同属性 -> 冲突；
        # 情境不重叠 -> 条件分支（ensure_branch 已按 scene 建分支）。
        context_overlap = _similarity(candidate.context, unit.context)
        value_similarity = (
            _similarity(candidate.value, branch.value) if (candidate.value and branch.value) else 0.0
        )
        contradiction = bool(
            candidate.attribute == unit.attribute
            and context_overlap > self.config.conflict_context_threshold
            and value_similarity < self.config.conflict_value_threshold
        )
```

（`if contradiction:` 及之后的逻辑保持不变。）

- [ ] **Step 4: 运行测试，确认通过**

Run: `cd /root/autodl-tmp/wangqihao/Mini-Agent-5-1 && python -m pytest tests/test_tpm_memory.py -v`
Expected: 全部 PASS。

- [ ] **Step 5: 提交**

```bash
cd /root/autodl-tmp/wangqihao
git add Mini-Agent-5-1/mini_agent/tpm/memory.py Mini-Agent-5-1/tests/test_tpm_memory.py
git commit -m "feat(tpm): configurable conflict thresholds + conditional-branch semantics (eq 10/11)

_fuse_candidate: contradiction = same-attribute AND context_overlap>δ_ctx AND
value_sim<δ_val (was hardcoded 0.35). Non-overlapping context -> conditional
scene branch instead of conflict.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 6: 心理导向 LLM 抽取 + 中文正则 + slot→memory_type 回填（论文式6、7）

LLM prompt/schema 改为心理子空间导向（slot 7 类 + memory_type 5 类 + relevance/utility）；正则补中文心理信号与 risk 识别；抽取器只给 slot 时按 §4.1 回填 $g_i$。

**Files:**
- Modify: `mini_agent/tpm/extractor.py`（`_build_payload` prompt/schema + `RegexProfileExtractor` 加中文 specs）
- Test: `tests/test_tpm_memory.py`

- [ ] **Step 1: 写失败测试 — 中文正则识别心理/风险信号 + slot 回填**

在 `tests/test_tpm_memory.py` 末尾追加：

```python
def test_regex_extractor_detects_chinese_psych_and_risk_signals():
    extractor = RegexProfileExtractor()

    risk = extractor.extract("我最近总是想死，觉得没意思", scene="general")
    assert any(c.slot == "risk" for c in risk)

    anxiety = extractor.extract("我最近很焦虑，压力大", scene="general")
    assert any(c.slot == "affect" for c in anxiety)
    assert any(c.slot == "stressor" for c in anxiety)


def test_llm_extractor_backfills_memory_type_when_only_slot_given():
    extractor = LLMProfileExtractor(
        api_key="dummy",
        api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model="deepseek-v4-flash",
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
                          "attribute": "stress",
                          "value": "工作压力很大",
                          "context": "用户描述工作压力",
                          "slot": "stressor",
                          "scene": "work",
                          "confidence": 0.88,
                          "stability": 0.6,
                          "relevance": 1.0,
                          "explicitness": 0.9,
                          "utility": 0.85
                        }
                      ]
                    }
                    """
                }
            }
        ]
    }
    with patch("mini_agent.tpm.extractor.requests.post", return_value=DummyHTTPResponse(payload)):
        candidates = extractor.extract("工作压力很大", scene="work")
    assert len(candidates) == 1
    assert candidates[0].slot == "stressor"
    assert candidates[0].memory_type == "stressor"  # §4.1 回填


def test_llm_extractor_prompt_requests_dual_fields():
    extractor = LLMProfileExtractor(
        api_key="dummy",
        api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model="deepseek-v4-flash",
    )
    payload = extractor._build_payload(text="我最近失眠", scene="general")
    user_msg = payload["messages"][-1]["content"]
    assert "slot" in user_msg
    assert "memory_type" in user_msg
    assert "relevance" in user_msg
    assert "utility" in user_msg
    assert "risk" in user_msg  # 心理/风险导向提示
```

- [ ] **Step 2: 运行测试，确认失败**

Run: `cd /root/autodl-tmp/wangqihao/Mini-Agent-5-1 && python -m pytest tests/test_tpm_memory.py::test_regex_extractor_detects_chinese_psych_and_risk_signals tests/test_tpm_memory.py::test_llm_extractor_backfills_memory_type_when_only_slot_given tests/test_tpm_memory.py::test_llm_extractor_prompt_requests_dual_fields -v`
Expected: FAIL（中文正则不存在；prompt 仍要 profile_type/recency/user_relevance）。

- [ ] **Step 3: 在 `RegexProfileExtractor` 增加中文心理/风险 specs**

在 `mini_agent/tpm/extractor.py` 的 `RegexProfileExtractor.extract` 的 `specs` 列表末尾（英文 specs 之后、`for pattern, ...` 循环之前）追加中文规则：

```python
            (r"\bfor this project[, ]+([^.,;!?]+)", "project_constraint", "behavior", "coping", 0.76, 0.62),
            # --- 中文心理/风险信号 ---
            (r"自伤|割腕|不想活|想死|自杀|轻生", "self_harm_risk", "risk", "affect", 0.95, 0.6),
            (r"我(?:很)?焦虑|焦虑症|恐慌", "anxiety", "affect", "affect", 0.85, 0.55),
            (r"抑郁|很丧|没(?:有)?动力|提不起劲", "depression", "affect", "affect", 0.82, 0.55),
            (r"压力(?:很|太)?大|压力大|喘不过气", "stress", "stressor", "stressor", 0.84, 0.58),
            (r"失眠|睡不着|睡不好|多梦", "sleep", "behavior", "affect", 0.8, 0.6),
            (r"我(?:叫|的名字是)\s*([^.,;!?]+)", "identity", "support", "trait", 0.9, 0.9),
            (r"我(?:喜欢|爱好)\s*([^.,;!?]+)", "interest", "behavior", "trait", 0.78, 0.7),
            (r"我(?:倾向|偏好)\s*([^.,;!?]+)", "style", "cognitive", "trait", 0.8, 0.78),
        ]
```

- [ ] **Step 4: 重写 `_build_payload` 为心理导向 prompt/schema**

把 `mini_agent/tpm/extractor.py` 的 `_build_payload` 方法整体替换为：

```python
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
```

- [ ] **Step 5: 运行测试，确认通过**

Run: `cd /root/autodl-tmp/wangqihao/Mini-Agent-5-1 && python -m pytest tests/test_tpm_memory.py -v`
Expected: 全部 PASS。

- [ ] **Step 6: 提交**

```bash
cd /root/autodl-tmp/wangqihao
git add Mini-Agent-5-1/mini_agent/tpm/extractor.py Mini-Agent-5-1/tests/test_tpm_memory.py
git commit -m "feat(tpm): psychology-oriented LLM extractor + Chinese regex + slot->g_i backfill

- LLM prompt/schema requests slot(7 psych subspaces)+memory_type(5)+relevance/utility
- Chinese regex for anxiety/stress/insomnia/depression/self-harm-risk + identity/interest/style
- _parse_candidates backfills memory_type via §4.1 when only slot is given

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 7: 历史感知抽取（论文式6、7：$f_{ext}(x_t, \mathcal{H}_{t-1})$）

`ProfileExtractor.extract` 增加 `recent_history` 入参；`begin_turn` 传入最近 `N=history_window` 轮用户消息；`agent.add_user_message` 从 `self.messages` 构造并喂入。

**Files:**
- Modify: `mini_agent/tpm/extractor.py`（`ProfileExtractor` 接口 + 两个实现的 `extract` 签名）
- Modify: `mini_agent/tpm/memory.py`（`TPMMemoryManager.begin_turn`）
- Modify: `mini_agent/agent.py`（仅 `add_user_message`）
- Test: `tests/test_tpm_memory.py`

- [ ] **Step 1: 写失败测试 — 历史被传入抽取器与 agent**

在 `tests/test_tpm_memory.py` 末尾追加：

```python
def test_begin_turn_forwards_recent_history_to_extractor():
    from mini_agent.tpm import TPMMemoryManager
    from mini_agent.tpm.extractor import ProfileExtractor

    class RecordingExtractor(ProfileExtractor):
        def __init__(self):
            self.seen_history = None

        def extract(self, text, scene="general", recent_history=None):
            self.seen_history = recent_history
            return []

    workspace = make_test_workspace()
    rec = RecordingExtractor()
    manager = TPMMemoryManager(memory_file=workspace / ".agent_memory.json", extractor=rec)
    manager.begin_turn("current message", scene="general", recent_history=["prev1", "prev2"])
    assert rec.seen_history == ["prev1", "prev2"]


@pytest.mark.asyncio
async def test_agent_passes_recent_user_history_to_begin_turn():
    from mini_agent.tpm import TPMMemoryManager
    from mini_agent.tpm.extractor import ProfileExtractor

    class RecordingExtractor(ProfileExtractor):
        def __init__(self):
            self.seen_history = None

        def extract(self, text, scene="general", recent_history=None):
            self.seen_history = recent_history
            return []

    workspace = make_test_workspace()
    rec = RecordingExtractor()
    manager = TPMMemoryManager(memory_file=workspace / ".agent_memory.json", extractor=rec)

    agent = Agent(
        llm_client=DummyLLM(),
        system_prompt="You are a helpful assistant.",
        tools=[],
        workspace_dir=workspace,
        memory_manager=manager,
    )
    agent.add_user_message("I like tea")
    agent.add_user_message("Tell me about tea")
    assert rec.seen_history is not None
    assert len(rec.seen_history) == 1
    assert "I like tea" in rec.seen_history[0]


def test_agent_caps_recent_history_to_window():
    from mini_agent.tpm import TPMMemoryManager
    from mini_agent.tpm.extractor import ProfileExtractor
    from mini_agent.tpm.memory import TPMConfig

    class RecordingExtractor(ProfileExtractor):
        def __init__(self):
            self.seen_history = None

        def extract(self, text, scene="general", recent_history=None):
            self.seen_history = recent_history
            return []

    workspace = make_test_workspace()
    rec = RecordingExtractor()
    manager = TPMMemoryManager(
        memory_file=workspace / ".agent_memory.json",
        extractor=rec,
        config=TPMConfig(history_window=2),
    )
    agent = Agent(
        llm_client=DummyLLM(),
        system_prompt="helper",
        tools=[],
        workspace_dir=workspace,
        memory_manager=manager,
    )
    for i in range(5):
        agent.add_user_message(f"msg {i}")
    assert len(rec.seen_history) == 2
```

- [ ] **Step 2: 运行测试，确认失败**

Run: `cd /root/autodl-tmp/wangqihao/Mini-Agent-5-1 && python -m pytest tests/test_tpm_memory.py::test_begin_turn_forwards_recent_history_to_extractor tests/test_tpm_memory.py::test_agent_passes_recent_user_history_to_begin_turn tests/test_tpm_memory.py::test_agent_caps_recent_history_to_window -v`
Expected: FAIL（`extract`/`begin_turn` 不接受 `recent_history`；agent 不传历史）。

- [ ] **Step 3: 给 `ProfileExtractor` 接口与两个实现加 `recent_history` 入参**

在 `mini_agent/tpm/extractor.py` 中：

(a) `ProfileExtractor.extract` 改为：

```python
    def extract(
        self, text: str, scene: str = "general", recent_history: list[str] | None = None
    ) -> list[ProfileCandidate]:
        raise NotImplementedError
```

(b) `RegexProfileExtractor.extract` 签名改为（扫描文本并入历史，提升跨轮信号捕获）：

```python
    def extract(
        self, text: str, scene: str = "general", recent_history: list[str] | None = None
    ) -> list[ProfileCandidate]:
        scan_text = "\n".join([*(recent_history or []), text])
        candidates: list[ProfileCandidate] = []
        specs = [
            # ... 保持现有 specs 不变 ...
        ]
        for pattern, attribute, slot, memory_type, confidence, stability in specs:
            for match in re.finditer(pattern, scan_text, flags=re.IGNORECASE):
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
```

（注意：`specs` 列表内容保持 Task 6 的完整版不变；`context=text` 用当前发言，`scan_text` 用于匹配。）

(c) `LLMProfileExtractor.extract` 签名改为接收并透传 `recent_history`：

```python
    def extract(
        self, text: str, scene: str = "general", recent_history: list[str] | None = None
    ) -> list[ProfileCandidate]:
        if requests is None or not self.api_key:
            return self._fallback(text, scene, recent_history=recent_history)

        try:
            payload = self._build_payload(text=text, scene=scene, recent_history=recent_history)
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
            return self._fallback(text, scene, recent_history=recent_history)

        return self._fallback(text, scene, recent_history=recent_history)
```

(d) `_build_payload` 签名增加 `recent_history` 并写入 prompt。把 `_build_payload(self, text, scene)` 改为 `_build_payload(self, text, scene, recent_history=None)`，并在 `user_prompt` 中 `f"最新用户发言：\n{text}\n\n"` 之后插入历史段：

```python
        history_block = ""
        if recent_history:
            joined = "\n".join(f"- {h}" for h in recent_history)
            history_block = f"最近用户发言历史（用于跨轮信号，非当前发言）：\n{joined}\n\n"
        user_prompt = (
            "任务：为 TPM 抽取心理画像候选。\n"
            f"当前场景：{scene}\n"
            f"{history_block}"
            f"最新用户发言：\n{text}\n\n"
            "抽取规则：\n"
            # ... 保持现有规则不变 ...
        )
```

(e) `_fallback` 签名增加 `recent_history` 并透传给 fallback 抽取器：

```python
    def _fallback(
        self, text: str, scene: str, recent_history: list[str] | None = None
    ) -> list[ProfileCandidate]:
        if self.fallback_extractor is None:
            return []
        return self.fallback_extractor.extract(text=text, scene=scene, recent_history=recent_history)
```

- [ ] **Step 4: `TPMMemoryManager.begin_turn` 接收并透传 `recent_history`**

把 `mini_agent/tpm/memory.py` 的 `begin_turn` 方法签名与调用改为：

```python
    def begin_turn(
        self,
        text: str,
        scene: str = "general",
        recent_history: list[str] | None = None,
    ) -> list[ProfileMemoryUnit]:
        self._load_from_disk()
        self._active_scene = scene
        self.memory.start_session(scene, session_id=self.session_id)
        candidates = self.extractor.extract(text, scene=scene, recent_history=recent_history)
        self.memory.ingest_candidates(candidates, scene=scene, session_id=self.session_id)
        self.memory.run_evolution_engine(scene, include_long_term_decay=False)
        self._save_to_disk()
        return self.memory.retrieve(text, scene=scene, top_k=self.retrieval_top_k)
```

- [ ] **Step 5: `agent.add_user_message` 构造并传入 recent history**

把 `mini_agent/tpm/../../agent.py`（`mini_agent/agent.py`）的 `add_user_message` 方法替换为：

```python
    def add_user_message(self, content: str, scene: str | None = None):
        """Add a user message to history."""
        message_content = content
        self.current_scene = scene or self.default_scene

        if self.memory_manager is not None:
            try:
                window = self.memory_manager.history_window
                prior_user = [
                    m.content
                    for m in self.messages
                    if m.role == "user" and isinstance(m.content, str)
                ]
                recent_history = prior_user[-window:] if window > 0 else []
                retrieved = self.memory_manager.begin_turn(
                    content, scene=self.current_scene, recent_history=recent_history
                )
                message_content = self.memory_manager.augment_user_message(content, retrieved)
                self._memory_turn_pending = True
            except Exception:
                message_content = content
                self._memory_turn_pending = False

        self.messages.append(Message(role="user", content=message_content))
```

（不触碰该方法之外的任何 LoRA 逻辑。）

- [ ] **Step 6: 运行测试，确认通过**

Run: `cd /root/autodl-tmp/wangqihao/Mini-Agent-5-1 && python -m pytest tests/test_tpm_memory.py -v`
Expected: 全部 PASS。

- [ ] **Step 7: 提交**

```bash
cd /root/autodl-tmp/wangqihao
git add Mini-Agent-5-1/mini_agent/tpm/extractor.py Mini-Agent-5-1/mini_agent/tpm/memory.py Mini-Agent-5-1/mini_agent/agent.py Mini-Agent-5-1/tests/test_tpm_memory.py
git commit -m "feat(tpm): history-aware extraction f_ext(x_t, H_{t-1}) (paper eq 6/7)

- ProfileExtractor.extract accepts recent_history; regex scans concatenated history, LLM includes it in prompt
- TPMMemoryManager.begin_turn forwards recent_history
- agent.add_user_message builds recent N=history_window user turns and passes them in
- LoRA logic untouched

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 8: 安全收尾 — 移除明文 API key + 环境变量回退（spec §9）

移除 `config.yaml:37` 明文 DeepSeek key，改环境变量 `DEEPSEEK_API_KEY` 读取。

**Files:**
- Modify: `mini_agent/config/config.yaml`（删除明文 key）
- Modify: `mini_agent/cli.py`（env 回退）
- Test: `tests/test_tpm_memory.py`

- [ ] **Step 1: 写失败测试 — key 不再明文 + env 回退**

在 `tests/test_tpm_memory.py` 末尾追加：

```python
def test_config_yaml_has_no_plaintext_deepseek_key():
    from mini_agent.config import Config

    leaked = "REDACTED_DEEPSEEK_KEY"
    cfg_path = Config.get_default_config_path()
    text = cfg_path.read_text(encoding="utf-8")
    assert leaked not in text


def test_extractor_api_key_env_fallback(monkeypatch):
    from mini_agent.cli import _extractor_api_key

    class FakeLLM:
        api_key = "fallback-llm-key"

    class FakeME:
        api_key = None  # yaml 已清空

    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-from-env")
    assert _extractor_api_key(FakeME(), FakeLLM()) == "sk-from-env"

    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    assert _extractor_api_key(FakeME(), FakeLLM()) == "fallback-llm-key"

    assert _extractor_api_key(type("M", (), {"api_key": "explicit"})(), FakeLLM()) == "explicit"
```

- [ ] **Step 2: 运行测试，确认失败**

Run: `cd /root/autodl-tmp/wangqihao/Mini-Agent-5-1 && python -m pytest tests/test_tpm_memory.py::test_config_yaml_has_no_plaintext_deepseek_key tests/test_tpm_memory.py::test_extractor_api_key_env_fallback -v`
Expected: FAIL（yaml 仍有明文 key；`_extractor_api_key` 未定义）。

- [ ] **Step 3: 移除 `config.yaml` 明文 key**

在 `mini_agent/config/config.yaml` 的 `memory_extractor:` 块中，把
`  api_key: "REDACTED_DEEPSEEK_KEY"`
改为：

```yaml
  api_key: ""   # 从环境变量 DEEPSEEK_API_KEY 读取，勿提交明文 key
```

- [ ] **Step 4: 在 `cli.py` 增加 `_extractor_api_key` 并用于三处构造**

在 `mini_agent/cli.py` 顶部确认 `import os` 存在（若无则添加）。在 `add_workspace_tools` 函数定义之前新增辅助函数：

```python
def _extractor_api_key(memory_extractor_cfg, llm_config) -> str | None:
    """解析抽取器 API key：显式配置 > 环境变量 DEEPSEEK_API_KEY > 主对话 key。"""
    primary = getattr(memory_extractor_cfg, "api_key", None)
    if primary:
        return primary
    return os.environ.get("DEEPSEEK_API_KEY") or getattr(llm_config, "api_key", None)
```

然后把 `add_workspace_tools` 中三处 `LLMProfileExtractor(...)` 构造里的 `api_key=...` 参数统一改为 `api_key=_extractor_api_key(memory_extractor_cfg, config.llm)`：

- 第一处（memory_extractor 分支）：`api_key=getattr(memory_extractor_cfg, "api_key", None) or config.llm.api_key,` → `api_key=_extractor_api_key(memory_extractor_cfg, config.llm),`
- 第二处（openai_compat 分支）：`api_key=getattr(config.llm, "openai_compat_api_key", None) or config.llm.api_key,` → `api_key=_extractor_api_key(memory_extractor_cfg, config.llm),`
- 第三处（provider 分支）：`api_key=config.llm.api_key,` → `api_key=_extractor_api_key(memory_extractor_cfg, config.llm),`

（注意：`memory_extractor_cfg` 在该函数中已定义；openai_compat/provider 分支里 `memory_extractor_cfg` 可能为 None，`_extractor_api_key` 用 `getattr(None, "api_key", None)` 安全返回 None，再走 env/主 key。）

- [ ] **Step 5: 运行测试，确认通过**

Run: `cd /root/autodl-tmp/wangqihao/Mini-Agent-5-1 && python -m pytest tests/test_tpm_memory.py -v`
Expected: 全部 PASS。

- [ ] **Step 6: 提交**

```bash
cd /root/autodl-tmp/wangqihao
git add Mini-Agent-5-1/mini_agent/config/config.yaml Mini-Agent-5-1/mini_agent/cli.py Mini-Agent-5-1/tests/test_tpm_memory.py
git commit -m "security(tpm): remove plaintext DeepSeek API key, use env var fallback

config.yaml memory_extractor.api_key cleared; cli resolves key via
explicit config > DEEPSEEK_API_KEY env > main LLM key.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 9: 全量回归与自审

**Files:** 无代码改动（仅校验）

- [ ] **Step 1: 全量测试**

Run: `cd /root/autodl-tmp/wangqihao/Mini-Agent-5-1 && python -m pytest tests/test_tpm_memory.py -v`
Expected: 全部 PASS（既有 10 个 + 新增约 16 个用例）。

- [ ] **Step 2: 导入冒烟测试（确保无循环导入/语法错误）**

Run: `cd /root/autodl-tmp/wangqihao/Mini-Agent-5-1 && python -c "from mini_agent.config import Config, build_tpm_config, TPMSettings; from mini_agent.tpm import TPMMemoryManager, TemporalProfileMemory, TPMConfig; from mini_agent.tpm.models import ProfileCandidate, ProfileMemoryUnit, migrate_profile_type, default_memory_type; from mini_agent.cli import _extractor_api_key; print('imports OK')"`
Expected: 输出 `imports OK`。

- [ ] **Step 3: 配置加载冒烟（实际 config.yaml 能解析）**

Run: `cd /root/autodl-tmp/wangqihao/Mini-Agent-5-1 && python -c "from mini_agent.config import Config; c=Config.load(); print('tpm.T_fresh=', c.tpm.T_fresh, 'history_window=', c.tpm.history_window)"`
Expected: 打印 `tpm.T_fresh= 168.0 history_window= 3`。

- [ ] **Step 4: 约束复核清单（人工确认）**

逐项核对：
- [ ] 未修改任何 `Table*/*_memory_bank.json`、`workspace/.agent_memory.json`、`Figure-data/` 数据文件（`git status` 不应出现这些路径的修改）。
- [ ] 未修改 `draft/TPPM-draft.tex`。
- [ ] 未触碰 `LoRA/`、`mini_agent/llm/local_lora_client.py`、`agent.py` 中 `_check_and_load_adapter` 与蒸馏子进程逻辑（`git diff Mini-Agent-5-1/mini_agent/agent.py` 仅 `add_user_message` 一处变动）。
- [ ] config.yaml 不含明文 `sk-` key。
- [ ] `from_dict` 仅只读兼容，无回写源文件逻辑。

Run: `cd /root/autodl-tmp/wangqihao && git diff --stat HEAD~8 -- Mini-Agent-5-1/` （核对改动文件范围仅限本计划清单）。

- [ ] **Step 5: 收尾提交（若有遗漏的文档/注释微调）**

```bash
cd /root/autodl-tmp/wangqihao
git status
# 若有未提交的相关改动：
git add -p Mini-Agent-5-1/
git commit -m "chore(tpm): post-alignment review touch-ups

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Self-Review（计划自审）

**1. Spec 覆盖核对:**
- §4 双字段数据模型 + §4.1/§4.2 映射 → Task 2（`DEFAULT_MEMORY_TYPE`/`LEGACY_PROFILE_TYPE_MAP`/`migrate_profile_type`/`default_memory_type`）✓
- §5.1 类型条件衰减 + 风险安全规则 R1 → Task 3 ✓
- §5.2 写入门控因子重命名（relevance/utility）→ Task 2 ✓
- §5.3 固化因子重命名（β2 evidence_strength→explicitness）→ 见下方修正 ⚠
- §5.4 冲突阈值 + 条件分支 → Task 5 ✓
- §6 Fresh+confidence 检索 → Task 4 ✓
- §7 心理导向抽取 + 历史感知 N=3 → Task 6 + Task 7 ✓
- §8 配置外置 → Task 1 ✓
- §9 安全（移除 key）→ Task 8 ✓
- §10 from_dict 只读兼容 → Task 2（迁移）✓
- §11 测试计划 → 分布于各 Task TDD + Task 9 ✓

**§5.3 修正（发现缺口）:** 固化因子 `evidence_strength → explicitness` 的局部变量重命名在 Task 2 未显式覆盖。需补：在 Task 2 Step 6 的 memory.py 修改中，把 `_promote_stable_memories` 里的 `evidence_strength = min(1.0, len(unit.evidence) / 4.0)` 重命名为 `explicitness = min(1.0, len(unit.evidence) / 4.0)`（计算不变，仅对齐主文 $X$ 命名），并把 `promote_weights[1] * evidence_strength` 改为 `promote_weights[1] * explicitness`。**此修正合并进 Task 2 Step 6 第 11 项之后追加一条修改。**

→ 已补入下方「Self-Review 修正补丁」。

**2. 占位符扫描:** 无 TBD/TODO/"add error handling"/"similar to Task N"。每个代码步骤均含完整代码。Task 7 Step 3(b) 的 `specs` 列表用注释 `# ... 保持现有 specs 不变 ...` 指代 Task 6 的完整列表——这是唯一一处「指代先前任务」的地方，为避免歧义，下方补丁给出明确指令：复用 Task 6 Step 3 完成后的 `specs` 列表全文，不得删减。

**3. 类型/命名一致性:** `slot`/`memory_type`/`is_risk`/`relevance`/`utility`/`history_window`/`T_fresh`/`conflict_context_threshold`/`conflict_value_threshold`/`_extractor_api_key`/`build_tpm_config`/`TPMSettings` 在所有 Task 中拼写一致。`ProfileExtractor.extract` 签名 `(text, scene, recent_history)` 在接口/Regex/LLM/`begin_turn`/agent 调用点一致。`TPMMemoryManager.history_window` 在 Task 1 定义、Task 7 使用，一致。

**4. Self-Review 修正补丁（合并进 Task 2）:**

在 Task 2 **Step 6** 末尾追加第 11 项修改：

> 11. `_promote_stable_memories` 中把 `evidence_strength = min(1.0, len(unit.evidence) / 4.0)` 重命名为 `explicitness = min(1.0, len(unit.evidence) / 4.0)`（对齐主文式12 的 $X$ 因子；计算不变），并把 `score = (... promote_weights[1] * evidence_strength ...)` 改为 `promote_weights[1] * explicitness`。

并在 Task 7 **Step 3(b)** 中明确：`specs` 列表必须包含 Task 6 Step 3 完成后的全部条目（英文 8 条 + 中文 9 条），逐字保留，不得省略。

# TPPM 代码对齐设计 — 按论文方法重构 Mini-Agent-5-1 的 `tpm` 模块

- 日期：2026-07-15
- 状态：已批准（决策 A1 / B / C；6 项默认全部按推荐；**数据暂不动**）
- 目标代码：`/root/autodl-tmp/wangqihao/Mini-Agent-5-1/mini_agent/tpm/` 及其接线
- 参考论文：`/root/autodl-tmp/wangqihao/draft/TPPM-draft.tex`（《时间心理画像记忆 TPPM》）

---

## 1. 背景与目标

`Mini-Agent-5-1` 是 MiniMax Mini-Agent 框架，其 `tpm/` 模块实现了论文提出的 TPPM（时间心理画像记忆）。前序分析发现：代码主干（三层记忆、写入门控、对齐/情境分支、状态–特质固化、类型衰减、多因子检索）与论文方法**高度对应**，附录超参数与 `TPMConfig` 几乎逐项吻合；但存在若干**代码↔论文不一致**与**论文自身矛盾**：

1. 画像类型是「通用用户画像」（goal/interest/style/background/preference/general），而非论文的心理子空间（情绪/压力/认知/应对/支持/节律/风险）；**风险信号安全规则未实现**。
2. 检索公式：论文主文式(16) 第4项=Fresh、第5项=q；附录与代码第4项=scene、第5项=quality；**Fresh 未计算**。
3. 写入/固化因子命名：主文 r(相关性)/X(显式度) ↔ 代码 recency/evidence_strength。
4. `δ_ctx` 在论文主文（冲突阈值）与附录（对齐阈值）用法矛盾；代码冲突判定用硬编码 0.35。
5. 候选抽取未使用历史 `H_{t-1}`；基座模型表述三处不一致（DeepSeek-V4-Flash / deepseek-v4-pro / Qwen3.5-9B 均为占位/不存在型号）。

**目标**：把代码按论文**主文方法**重构对齐，使 `tpm` 模块成为论文方法的忠实实现。保持 TPPM 规则驱动（与论文"无可训练参数"一致）；**LoRA 部分不在本次范围内**（按要求不碰）。

## 2. 锁定决策

- **A1 全量心理域对齐**：含 `slot`/`memory_type` 双字段重构 + 持久化向后兼容。
- **B 拆双字段**：`profile_type` 拆为 `slot`($a_i$) + `memory_type`($g_i$)。
- **C 以主文为准**：检索按主文式(16)（Fresh + confidence）；写入/固化因子按主文命名。
- **数据暂不动**：不修改任何现有数据文件；`from_dict` 仅只读向后兼容。
- 风险反证机制：**R1**（复用 contradiction 路径）。
- `T_fresh=168h`；历史窗口 `N=3`；`touch_access` 副作用耦合保留（仅文档说明）。
- 不提供 `Table*/*_memory_bank.json` 批量迁移脚本。

## 3. 范围

**改**：`tpm/models.py`、`tpm/memory.py`、`tpm/extractor.py`、`agent.py`、`cli.py`、`config.py`、`config/config.yaml`、`tests/test_tpm_memory.py`。

**不改**：`LoRA/` 全部、`tools/note_tool.py`（接口不变，自动受益）、`skills/`、`mcp/`、`acp/`、`schema/`、`utils/`。

**不触碰**：任何现有数据文件（`Table*/*_memory_bank.json`、`workspace/.agent_memory.json`、`Figure-data/` 等）。

**保持**：TPPM 规则驱动、三层记忆主干、情境分支结构、证据集合、`SceneProfileBranch` 结构。

**Out of scope（建议另起任务）**：修订论文 TeX 附录使之与主文/代码一致（附录现写 recency/历史证据量/scene/quality，与主文矛盾）；对齐 `Table*/` 评测脚本中对 `profile_type` 字段的读取（若其依赖旧字段名）。

## 4. 数据模型：`slot` + `memory_type` 双字段

**文件**：`tpm/models.py`

- `ProfileCandidate`、`ProfileMemoryUnit`：`profile_type` 拆为
  - `slot` ∈ {`affect, stressor, cognitive, coping, support, behavior, risk`}（论文表1 的 7 个心理子空间，即 $a_i$）
  - `memory_type` ∈ {`affect, stressor, coping, support, trait`}（论文式15 的 5 个时间–安全类型，即 $g_i$，索引 $\lambda_{g_i}$）
- `ProfileMemoryUnit` 增派生属性 `is_risk = (slot == "risk")`，驱动风险安全规则。
- `EvidenceItem`、`SceneProfileBranch` 结构不变。
- `to_dict`：写新双字段 schema（不再写 `profile_type`）。
- `from_dict`：**只读向后兼容**——读取旧 `profile_type` 时按下表迁移为 `{slot, memory_type}`（仅内存，不回写源文件）；同时支持新 schema 直读。

### 4.1 默认 slot → memory_type 映射（抽取器只给 slot 时回填 $g_i$）

| slot | memory_type | 理由 |
|---|---|---|
| affect | affect | 情绪最易变 |
| stressor | stressor | 压力源次之 |
| coping | coping | 应对方式中等 |
| support | support | 社会支持中等 |
| cognitive | trait | 认知/信念较稳定 |
| behavior | coping | 行为节律中等 |
| risk | affect | risk 单元由 `is_risk` 覆盖衰减，$g_i$ 仅作记录 |

衰减率满足论文式(15) 排序：$\lambda_{\text{affect}}>\lambda_{\text{stressor}}>\lambda_{\text{coping}}\approx\lambda_{\text{support}}>\lambda_{\text{trait}}$。

### 4.2 旧 `profile_type` → `{slot, memory_type}` 迁移表（best-effort，仅 `from_dict` 内存解析）

| legacy | slot | memory_type |
|---|---|---|
| background | support | trait |
| preference | cognitive | trait |
| goal | behavior | coping |
| style | cognitive | trait |
| interest | behavior | trait |
| general | coping | trait |

> legacy `general` 语义模糊，默认 `{coping, trait}`；如需精确可在加载后人工重标。该映射为只读解析，**不修改源文件**。

## 5. 演化机制对齐

**文件**：`tpm/memory.py`、`tpm/models.py`

### 5.1 类型条件衰减 + 风险安全规则（论文式14、15）
- `TPMConfig.decay_lambdas` 键改 $g_i$：`{affect:0.10, stressor:0.07, coping:0.05, support:0.05, trait:0.03}`。
- `decay_long_term()`：`if unit.is_risk:` **跳过** `s·exp(−λΔt)` 常规衰减；仅当该 risk 单元在 `_fuse_candidate` 命中 contradiction 时，由 `−γ⁻·ψ⁻` 降低强度（$\psi^-$ = contradiction 信号，复用 `contradiction_count`）。
- 逐轮衰减（working/short_term，`_decay_store`）+ 会话末衰减（long_term，`decay_long_term`）现状已符合论文，保留。

### 5.2 写入门控因子命名（论文式8）
- `ProfileCandidate.write_score`：因子 `(recency, explicitness, user_relevance, stability)` → 重命名为 `(relevance, explicitness, utility, stability)`（对齐主文 $r/e/u/b$）。权重 `(0.25, 0.3, 0.25, 0.2)` 不变。
- `ProfileCandidate` 字段 `recency` → `relevance`；`user_relevance` → `utility`。

### 5.3 状态–特质固化因子命名（论文式12）
- `_promote_stable_memories` 的 $\Pi$ 因子：β2 由 `evidence_strength`(证据量) → `explicitness`(显式度)，对齐主文 $X$。其余 $R$=reinforcement、$U$=utility、$S$=stability、$C$=contradiction 对齐命名。`promote_weights=(0.35,0.2,0.15,0.25,0.2)` 不变。

### 5.4 对齐 / 分支 / 冲突阈值（论文式10、11 + 主文 δ_ctx）
- `context_threshold=0.62` 保留为**对齐匹配阈值**（相似度低于它 → 新建单元/分支，与附录一致）。
- 新增可配项 `conflict_context_threshold`(≈δ_ctx=0.62) 与 `conflict_value_threshold`(≈0.35)；`_fuse_candidate` 冲突判定从硬编码 0.35 改为配置，实现主文语义：**情境重叠 $\rho>\delta_{\text{ctx}}$ 且极性相反 → 冲突更新；情境不重叠 → 建立条件分支**。

## 6. 画像感知检索（主文式16、17）

**文件**：`tpm/memory.py`（`_retrieve_score`、`retrieve`、`TPMConfig.retrieve_weights`）

- 5 因子：`η1·Rel + η2·stability + η3·Ctx + η4·Fresh + η5·confidence`
- `Rel = max(sim(query, value), sim(query, context))`（把原 `ctx_score` 文本相似度并入相关性）。`Ctx` = **场景匹配**（现 `w4·scene_score` 上移到第3位）。
- 新增 `Fresh(unit) = exp(−Δt / T_fresh)`，`Δt` 取 `last_accessed` 距今小时数；`T_fresh=168h`（可配）。
- 第5因子由 `quality_score` 改为 `confidence_score`($q$)。
- `retrieve_weights` 重命名 `(rel, stability, ctx, fresh, confidence)`，权重 `(0.35, 0.2, 0.15, 0.2, 0.1)`。
- `retrieve()` 中 `unit.touch_access()` 副作用仍影响固化（`usage_strength`），保留并在代码注释/文档说明该耦合。

## 7. 候选抽取：心理导向 + 历史感知（论文式6、7）

**文件**：`tpm/extractor.py`、`tpm/memory.py`（`begin_turn`）、`agent.py`（`add_user_message`）

- `LLMProfileExtractor` prompt/schema：输出 `slot`(7类)+`memory_type`(5类)，抽取规则改为情绪/压力/认知/应对/支持/节律/风险导向；`profile_type` 校验集合替换为 slot+memory_type 校验；评分字段同步重命名为 `relevance`/`utility`（见 §5.2）。
- `RegexProfileExtractor`：补**中文心理信号**正则（焦虑/压力大/失眠/自伤念头等）+ risk 识别，使中文回退可用。
- **历史感知**：`TPMMemoryManager` 增加 `recent_history` 入口；`begin_turn` 除当前 `text` 外传入最近 `N=3` 轮用户消息，落地 $f_{\text{ext}}(x_t, \mathcal{H}_{t-1})$；`agent.add_user_message` 喂入。

## 8. 配置外置

**文件**：`config/config.yaml`、`config/config.py`、`cli.py:537`（`add_workspace_tools`）

- `config.yaml` 新增 `tpm:` 块：thresholds（write/context/promote/promotion_min_sessions/conflict_*）、weights（write/promote/retrieve）、decay_lambdas、risk、`T_fresh`、`history_window`。
- `config.py` 解析为 `TPMConfig`；`cli.py:add_workspace_tools` 构造 `TPMMemoryManager(memory_file=..., extractor=..., config=tpm_config, retrieval_top_k=...)`。
- `TPMConfig` 现有默认值作为配置缺失时的回退。

## 9. 模型表述与安全

- 基座模型表述统一：**需你确认实验实际使用的基座模型**（代码侧 `local_lora` 路径指向 `Qwen2.5-7B-Instruct`，抽取器可配置独立 API 模型）。确定后，`config.yaml` 与论文附录统一为该真实型号，替换占位名 "DeepSeek-V4-Flash / deepseek-v4-pro / Qwen3.5-9B"。
- 移除 `config.yaml:37` 明文 DeepSeek API key（`sk-abdc93...`），改环境变量读取。

## 10. 持久化迁移与兼容

- `from_dict`：**只读向后兼容**，按 §4.2 把旧 `profile_type` 解析为双字段；支持新 schema 直读。**不回写、不迁移任何源文件**。
- **不提供** `Table*/*_memory_bank.json` 批量迁移脚本；不修改任何现有数据文件。
- 运行时 `workspace/.agent_memory.json`：agent 下次运行时会以新 schema 保存（新格式是旧格式的超集，**无损加字段**）；建议首次运行前备份该文件。
- LoRA 蒸馏器（`LoRA/tppm_distiller.py`）读取 `long_term_memory` 的 `strength`/`confidence` 两键，双字段重构不改这两键 → 不影响蒸馏链路（且 LoRA 不在范围内）。
- 兼容性风险：若有其它代码（如 `Table*/` 评测脚本）直接读 `profile_type` 字段，会在新 schema 下失效 → 列为 out-of-scope 风险，需另行对齐。

## 11. 测试计划

**文件**：`tests/test_tpm_memory.py`（新增/更新）

- 风险单元：`is_risk=True` 时 `decay_long_term` 不随时间下降；命中 contradiction 后才降强度。
- Fresh 检索：相同语义下 `last_accessed` 更新的单元排名更高。
- 冲突阈值：情境重叠+极性相反 → 冲突更新；情境不重叠 → 建立分支（用 `conflict_*` 配置驱动）。
- 双字段迁移：旧 `profile_type` JSON 经 `from_dict` 后 `slot`/`memory_type` 正确；新 schema 往返幂等。
- 历史感知抽取：`begin_turn` 传入 `recent_history` 后抽取结果包含跨轮信号。
- slot→memory_type 默认回填：抽取器只给 slot 时 $g_i$ 按 §4.1 回填。
- 回归：写入门控、对齐、固化、情境分支、证据集合既有行为不破坏。

## 12. 改动文件清单

| 文件 | 章节 | 改动要点 |
|---|---|---|
| `tpm/models.py` | 4,5.2,10 | slot/memory_type 双字段、is_risk、write_score 因子重命名、from_dict 迁移、to_dict 新 schema |
| `tpm/memory.py` | 5,6,7,10 | decay_lambdas+风险规则、检索 Fresh/confidence、冲突阈值、固化因子、历史抽取入口、from_dict 迁移 |
| `tpm/extractor.py` | 7 | 心理导向 prompt/schema、中文正则、slot+type 输出、历史入参 |
| `agent.py` | 7 | `add_user_message` 传入 recent history |
| `cli.py` | 8 | 读取 tpm 配置、传入 TPMConfig |
| `config.py` / `config.yaml` | 8,9 | tpm 配置块、基座模型统一、移除 key |
| `tests/test_tpm_memory.py` | 11 | 风险/Fresh/冲突/迁移/历史抽取用例 |

## 13. 实施顺序（建议）

1. §8 配置外置（打通可调参通路）。
2. §5.1 风险规则 + §6 Fresh 检索（机制补齐，可独立验证）。
3. §4 双字段数据模型 + §5.4 冲突阈值 + §5.2/5.3 因子命名。
4. §7 心理导向抽取 + 历史感知。
5. §10 from_dict 兼容 + §11 测试。
6. §9 模型表述/安全收尾。
7. 每阶段跑 `pytest tests/test_tpm_memory.py` 回归。

## 14. 风险与缓解

- **数据兼容**：双字段改变 schema。缓解：`from_dict` 只读兼容旧格式；不迁移任何现有数据；运行时保存为无损超集。评测脚本若读 `profile_type` 需另行对齐（out of scope）。
- **论文自洽**：主文与附录在检索公式、δ_ctx、因子命名上矛盾。本次以主文为准实现代码；附录修订建议另起任务，否则论文仍内部不一致。
- **slot→type 迁移主观性**：legacy 映射为 best-effort，`general` 等模糊项默认值可能不准；仅影响旧数据加载后的标签，可人工重标。
- **风险规则 R1 的反证判定**：以 contradiction 作为安全反证是简化；若需"明确安全证据"语义，后续可升级为 R2（`EvidenceItem.safety_counter` 标签）。
- **耦合**：`retrieve()` 的 `touch_access` 副作用影响固化，保留并文档化。

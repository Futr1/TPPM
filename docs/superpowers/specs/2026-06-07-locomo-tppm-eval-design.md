# LoCoMo TPPM 评估 — 实验一 Layer 2 设计文档

**日期**：2026-06-07
**状态**：已批准
**主题**：基于 Mini-Agent-5-1 TPPM 引擎，在 LoCoMo 数据集上实现实验一 Layer 2 评估流水线（QA + Event Summarization）

---

## 1. 背景

实验一 Layer 2 在 LoCoMo 的 10 段超长对话（平均 ~600 轮 / ~16K tokens / 最多 32 sessions）上评估 TPPM 的跨会话记忆保持能力。评估协议与 Maharana et al. 2024 Table 2/3/4 完全对齐。

## 2. 架构：三阶段流水线

```
阶段 1（离线，一次性）                    阶段 2a（评估，可反复运行）
┌────────────────────────────┐      ┌──────────────────────────────┐
│ locomo_tppm_extract.py     │      │ locomo_qa_eval.py            │
│                            │  →   │                              │
│ 输入: locomo10.json        │      │ 加载 memory bank             │
│ 范围: 10 段对话 × N session │      │ 混合式上下文构建             │
│ API: DeepSeek (async×8)    │      │ vLLM Qwen3.5-9B 生成答案     │
│ 引擎: TemporalProfileMemory│      │ token-level F1（官方 evalu）  │
│ 参数: TPMConfig（通用版）   │      │                              │
│                            │      │ 产物: locomo_qa_results.json │
│ 产物: locomo_memory_bank   │      └──────────────────────────────┘
│       .json                │
└────────────────────────────┘      阶段 2b（评估，可反复运行）
                                    ┌──────────────────────────────┐
                                    │ locomo_event_eval.py         │
                                    │                              │
                                    │ 加载 memory bank             │
                                    │ 时间范围 → 检索记忆          │
                                    │ vLLM Qwen3.5-9B 生成事件     │
                                    │ FactScore + ROUGE-L          │
                                    │                              │
                                    │ 产物: locomo_event_results   │
                                    │       .json                  │
                                    └──────────────────────────────┘
```

## 3. 上下文构建策略（混合式 C）

```
System Prompt = 基础指令 + TPPM 结构化画像

Messages = [
    早期 session 的 session_summary（LoCoMo 已有，压缩历史），
    最近 3 个 session 的完整对话原文，
    QA 问题 / 事件摘要指令
]
```

TPPM 画像通过结构化信息弥补截断历史丢失的关键参照，最近 3 session 保留完整上下文确保局部对话质量。

## 4. 评估协议对齐

### QA（5 类推理）

| 类别 | 编号 | 评估方法 |
|------|------|---------|
| Multi-hop | 1 | 逗号拆分子答案，逐项 F1 取均值 |
| Single-hop | 2 | Token-level F1（Porter stem + token set） |
| Temporal | 3 | 取分号前首答案，Token-level F1 |
| Open-domain | 4 | Token-level F1 |
| Adversarial | 5 | 检测 "no information available" / "not mentioned" |

完全复用 LoCoMo 官方 `task_eval/evaluation.py` 的 `f1_score()` / `f1()` / `eval_question_answering()`。

### Event Summarization

- **主指标**：FactScore P/R/F1（原子事实分解 + 蕴含判断）
- **辅助指标**：ROUGE-1/2/L

## 5. 从 Mini-Agent-5-1 复用的模块

| 模块 | 用途 |
|------|------|
| `mini_agent/tpm/memory.py` → `TemporalProfileMemory` + `TPMConfig` | 完整记忆引擎 |
| `mini_agent/tpm/models.py` → `ProfileMemoryUnit` 等 | 数据模型 |
| `mini_agent/tpm/extractor.py` → `LLMProfileExtractor` | DeepSeek API 抽取 |
| `mini_agent/LoRA/tppm_teacher_distiller.py` → async 并发模式 | 8 并发控制 |

## 6. TPPM 参数（TPMConfig 通用版）

- write_threshold=0.68, context_threshold=0.62, promote_threshold=0.72
- write_weights=(0.25, 0.3, 0.25, 0.2)
- decay_lambdas: {goal:0.1, interest:0.07, style:0.04, background:0.04, preference:0.05, general:0.05}

## 7. 输出文件

| 文件 | 路径 |
|------|------|
| Memory Bank | `Table2-data/outputs/locomo_memory_bank.json` |
| QA Results | `Table2-data/outputs/locomo_qa_results.json` |
| Event Results | `Table2-data/outputs/locomo_event_results.json` |
| 失败日志 | `Table2-data/logs/locomo_*.jsonl` |

## 8. 基线数据来源

LoCoMo 已评测 11 个基线（Base LLM ×3 + Long-context ×4 + RAG×3 + Human），数字直接引用 Maharana et al. 2024 Table 2/3/4，无需重新运行。

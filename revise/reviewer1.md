# 审稿人 #1 回复

---

## 意见 1

> 本文提出了一种新颖的混合路径规划框架，结合了 IFDS 的局部避障能力与 MPC 的全局优化特性，在理论创新方面表现突出，尤其是对光影干扰这类动态"软障碍"的适应性。然而，IFDS 的流体动力学调制与 MPC 的滚动时域优化之间的协同机制阐述不够清晰。请作者进一步说明两者在控制决策中的主次关系与交互逻辑，特别是多约束条件下的优先级调度机制。

---

### 回复

感谢审稿人的建设性意见。我们同意原稿未充分阐述 IFDS 与 MPC 之间的协同机制及约束优先级层次。我们做出了以下修改：

**1. 重写了第 3 节（Methodology）开篇段落**

原段落仅以叙事方式罗列各小节结构。我们将其替换为完整的控制架构总览，明确建立了 **"IFDS 生成，MPC 优化"** 的层级范式：

- **IFDS（路径生成器）：** 将无人机航点视为流体粒子，在每个阴影障碍物（建模为超椭球体）周围构建排斥调制场，生成平滑、无碰撞的候选速度场，使其像流体绕流一样自然偏转绕过干扰区域。

- **MPC（参数优化器）：** 在每个控制周期内，枚举 IFDS 调制场的候选参数配置（速度归一化因子、排斥系数、DFAA 下降增益），通过复合代价函数评估各候选轨迹，利用滚动时域优化选择最优配置。

这种分工——IFDS 提供具有物理基础的结构化搜索空间，MPC 在其内部执行多目标优化——是在动态环境中实现高效实时路径规划的关键。

**2. 明确定义了分层约束优先级**

修改后的文本明确区分了三个层级：

| 层级 | 类型 | 内容 | 机制 |
|:---:|:---:|:---|:---|
| **L1** | 硬约束 | 障碍边界不可穿透 $\Gamma(P) > 1$、飞行动力学限制（最大转弯速率、航迹倾角、飞行高度） | 不可行候选路径直接丢弃（$J = \infty$），安全绝不妥协 |
| **L2** | 软优化 | 跟踪精度、路径平滑度、观测覆盖 | 通过代价函数权重 $\lambda_1$–$\lambda_3$ 调节 |
| **L3** | 条件触发 | 有效观测宽度 $W_{eff} < \tau$ 时触发 DFAA 高度调节 | 嵌入 IFDS 调制矩阵，仅在 L1 满足时激活 |

L1 安全约束绝不会被妥协——在任何软优化执行之前，通过硬可行性过滤（$J = \infty$）强制执行。L2 目标通过可调权重实现 Pareto 平衡。L3 DFAA 机制直接扰动调制矩阵以获得低延迟响应，但受 L1 安全条件门控。

**3. 明确了交互逻辑**

每个控制周期（$\Delta T = 0.1$s）内的执行流程：
1. EKF 预测未来 $N$ 步的阴影障碍状态
2. IFDS 枚举候选参数集并生成对应的候选速度场
3. MPC 以复合代价函数评估所有候选路径并选择最优
4. 执行最优路径的第一步，优化窗口向前滚动

DFAA 机制有意嵌入在 IFDS 内部（而非作为 MPC 的代价项），因为可观测宽度的突然变化需要确定性的低延迟响应——将其作为 MPC 优化变量会增加优化问题维度，可能违反实时性约束。

**4. 在 3.4 节添加了衔接 IFDS 输出与 MPC 输入的桥接句**

原稿从 IFDS 调制（3.3 节）直接跳入 MPC 滚动优化（3.4 节），中间缺乏衔接。我们在 3.4 节开头增加了一句桥接，明确指出 IFDS 产生的调制速度场构成候选轨迹族，MPC 从中评估和选择。

**5. 添加了对系统架构图（图 1）的交叉引用**

开篇句现在显式引用了系统架构图，为读者理解模块交互提供了可视化锚点。

---

### 稿件中修改后的文本

**重写的开篇段落（第 3 节）：**

> The proposed control framework, illustrated in Figure 1, follows a hierarchical "IFDS generates, MPC optimizes" paradigm. At its core, the Improved Interfered Fluid Dynamical System (IFDS) serves as the path generator: it treats UAV waypoints as fluid particles and constructs a repulsive modulation field around each shadow obstacle (modeled as a super-ellipsoid), producing smooth, collision-free candidate velocity fields that naturally deflect around disturbance zones. The Model Predictive Control (MPC) layer functions as the parameter optimizer: within each control cycle, it enumerates candidate parameter configurations of the IFDS modulation field—including the velocity normalization factor, repulsion coefficient, and DFAA descent gain—evaluates the resulting candidate trajectories through a composite cost function, and selects the optimal configuration via receding-horizon optimization. This division of labor—IFDS providing a physically-grounded, structured search space and MPC performing multi-objective optimization within it—enables efficient real-time path planning in dynamic environments.
>
> The framework enforces a tiered constraint hierarchy. Hard constraints, including obstacle boundary non-penetration ($\Gamma(P) > 1$) and flight dynamics limits (maximum turn rate, flight path angle, and altitude bounds), are guaranteed by immediately discarding infeasible candidate paths ($J = \infty$), ensuring that safety is never compromised by soft optimization objectives. Soft objectives—tracking fidelity, path smoothness, and observation coverage—are balanced through adjustable cost function weights ($\lambda_1$–$\lambda_3$) within the MPC layer. When the effective observable width falls below a mission-defined threshold and all safety constraints remain satisfied, the Dynamic Flight Altitude Adjustment (DFAA) mechanism, embedded within the IFDS modulation matrix, directly perturbs the velocity field in the vertical direction to prioritize observation resolution. The remainder of this section details each module: dynamic obstacle modeling (Section 3.1), initial path and velocity field construction (Section 3.2), IFDS modulation with DFAA (Section 3.3), and MPC-based rolling optimization (Section 3.4).

**3.4 节桥接句：**

> The modulated velocity fields produced by the IFDS layer (Section 3.3), each corresponding to a specific configuration of the modulation parameters, constitute a family of candidate trajectories. To select the optimal among these candidates, this study adopts a Rolling Optimization [...]

---

### 修改摘要

| # | 修改项 | 位置 | 修改前 | 修改后 |
|:---:|:---|:---|:---|:---|
| 1 | 重写 3. Methodology 首段 | `.tex` L177–179 | "In this section... first... Next... Finally..." 纯叙事罗列 | "IFDS 生成，MPC 优化" 范式 + 三层约束优先级 |
| 2 | 3.4 节添加桥接句 | `.tex` L285 | 无衔接，直接进入 MPC 描述 | IFDS 调制速度场构成候选轨迹族，MPC 从中择优 |
| 3 | 添加子节标签 | `.tex` L182/227/253/284 | 无 `\label` | `sec:obstacle-modeling`、`sec:initial-path`、`sec:ifds`、`sec:mpc` |
| 4 | Section 引用改为 `\ref{}` | `.tex` L179 | 硬编码 "Section 3.1" | `Section~\ref{sec:obstacle-modeling}` |
| 5 | 图片路径统一到 `Figures/` | `.tex` 10处 | `fig1.PDF`（散落主目录） | `Figures/fig1.PDF`（集中在 Figures 目录） |

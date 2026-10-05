# 2026-10 阶段任务报告（文件清单）

## 一、本文件说明

本文档只记录 2026-10 阶段创建/复制的文件清单，以及每个文件里有哪些东西。研究内容（研究问题、计划、组会汇报等）见 `Research_Goal_2026Oct.tex`。

## 二、文件清单

### 1. Research_Goal_2026Oct.tex（主文档）

内容结构：

- preamble（IEEEtran 文档类 + ctex 中文支持 + 输入 `math_commands_TESS.tex` / `math_commands_STLLR.tex` + 颜色定义）
- title、abstract
- 文章汇总（6 篇文献引用：RTRL / e-prop / OTTT / NDOT / S-TLLR / TESS）
- 研究背景与核心问题（本地学习缺少全局梯度、核心研究问题）
- 三个研究方向（逐层探针 / 逐层表示相似度 / 逐层梯度对齐 + 三者关系）
- 已完成工作（研究计划文档 / Project 基础代码 / PPT 笔记推导 / 冻结关键约定）
- 下一步计划（阶段 0–6）
- 组会报告汇总（2026-10-08 详细 + 2026-10-XX / 2026-11-XX 占位）
- 参考文献（`SNN_OnlineLearningSummary.bib`）

### 2. Task_Report.md（本文件）

内容结构：

- 本文件说明
- 文件清单

### 3. SNN 在线本地学习：细节问题清单.md（详细版）

内容结构：

- 一、已解决（含详细公式推导）：符号与约定 / 资格迹 / 更新时机 / 算法理解 / 时空解耦 / 空间局部化 / 三因子规则 / 信息量化与资源量化 / 正定矩阵（详细）/ 收敛性证明与数值细节（本次阅读论文后新增：OTTT Theorem 1、NDOT clamp、TESS B 准正交、LIF 复位形式比较、资格迹粒度理论）
- 二、待解决：需精读原文（疑问 C：DFA 收敛证明）/ 需实验解决（疑问 F：e-prop 反馈、疑问 G 实验部分、H/I/J）
- 三、关键约定汇总表（已冻结）

### 4. 编译支撑文件（从 2026_09 复制）

- `math_commands_TESS.tex` —— TESS 的数学命令定义
- `math_commands_STLLR.tex` —— S-TLLR 的数学命令定义
- `SNN_OnlineLearningSummary.bib` —— 参考文献（6 篇）
- `IEEEtran.cls` —— IEEEtran 文档类
- `IEEEtran.bst` —— IEEEtran 参考文献样式

### 5. Project/MANUAL.md（项目手册，本次新增）

内容结构：

- 项目现状盘点（已实现内容对应阶段、阶段 0 剩余任务、阶段 1 需新建文件、文件→阶段→状态表）
- 目录结构（目录树）
- 逐文件说明（每个文件的路径 / 作用 / 函数 / 依赖 / 对应阶段）
- 逐函数说明（每个函数的签名 / 参数 / 返回 / 公式 / 使用示例 / TODO）
- 未实现文件清单（数据模块、train_epoch、evaluate、bptt_sg、gradient_check）
- 数学符号与代码对应表（λ→lambd、V_th→v_th 等）

> 总说明：Project 下所有 `.py` 文件的函数/类已补 docstring 与关键行内注释（不改逻辑、不改签名）；surrogate 接 autograd 已实现（`SpikeFunction` + `LIF.step` 接入，`BETA` 新增）；阶段 0 框架已完成（数据编码 `data/`、训练循环 `utils/train.py`、指标接口 `utils/evaluate.py`、验证 `sanity/stage0_check.py`）。

## 三、组会记录

见 `Research_Goal_2026Oct.tex` 的"组会报告汇总" section。

# SNNInformationReconstruction

SNN训练中 如何在约束条件下恢复被丢失的信息 的问题

---

## 项目结构

### Algorithm/
SNN在线/本地学习算法相关论文与代码。包含 E-prop、NDOT、OTTT、RTRL、S-TLLR、TESS 等算法的论文PDF及转换后的文本。

### GroupMeeting&CourseSlide/
组会汇报与课程幻灯片的 LaTeX 源文件及编译产物。主体文件为 `SNN_OnlineLearningSummary.tex`，涵盖以下四个算法的详细论述与数学推导：
- **BPTT（基础）**：完整的时间展开梯度公式、代理梯度（SG）与 STBP 的严格定义、核心瓶颈分析
- **OTTT**：通过丢弃重置路径实现 O(1) 内存的在线训练，时序核退化为固定 λ 衰减
- **NDOT**：基于神经元动力学重参数化，以动态比值 e[t] 替代固定 λ，在相同内存下提升精度
- **S-TLLR**：受 STDP 启发的三因子学习规则，融合因果与非因果时序关系，实现时间本地化
- **TESS**：在 S-TLLR 基础上通过 LSG（Local Signal Generation）进一步实现空间本地化，达到时空全本地学习

该文档为 LaTeX + IEEEtran 格式，包含讲稿（PPT内容要点 + 口头讲稿），已编译为 PDF。这一部分为重点部分，为主要工作成果结晶，配合组会PPT进行阅读。

### Work_Copilot_Drafts/
一系列工作草稿文件，大部分为 vibe_code 内容，包含：
- `Draft.md`：项目初始思路草稿——关于各算法中信息丢失与恢复的核心思想
- `Algorithm_Summaries/`：OTTT 与 NDOT 算法的单独详细总结（LaTeX + 编译PDF）
- `Meeting_Report/`：会议报告草稿
- `Notes/`：大型笔记文档（`SNNInformationReconstruction.tex`，约 1457 行），涵盖 SNN 信息重建的全面笔记
- `Total_Summary/`：总结与对比分析，包括：
  - `对比分析.md`：OTTT vs NDOT vs S-TLLR vs TESS 四种算法的详细对比（公式对比、精度对比、特性矩阵）
  - `snn_info_summary_cn.tex`：SNN 训练中信息丢失与恢复的中文汇总与研究思路
  - `snn_info_summary.tex`：英文版汇总
  - `SNN中的时间步及其训练机制/`：SNN 时间步相关子专题

---

## 更新日志

### 2026-06-30
- **GroupMeeting&CourseSlide/**：完成 `SNN_OnlineLearningSummary.tex`，对 BPTT、OTTT、NDOT、S-TLLR、TESS 五个算法进行了详细论述与数学推导，包含 PPT 内容要点和讲稿。从 BPTT 的标准梯度公式出发，逐步推导至 OTTT 的近似解耦（O(1) 内存在线学习）、NDOT 的物理重参数化（动态 e[t] 替换固定 λ）、S-TLLR 的 STDP 启发三因子规则（时间本地化）、TESS 的 LSG 机制（时空全本地化）。梳理了完整的算法演化路线：BPTT → OTTT → NDOT → S-TLLR → TESS。RTRL 与 E-prop 部分暂缓。
- **Work_Copilot_Drafts/**：维护了大量草稿与 vibe_code 内容，其中 `Total_Summary/对比分析.md` 完成了四种算法的详细横向对比（精度矩阵、特性矩阵、一句话总结），`Total_Summary/snn_info_summary_cn.tex` 汇总了信息丢失与恢复的研究思路，`Notes/SNNInformationReconstruction.tex` 积累了约 1457 行的全面笔记。Algorithm_Summaries/ 下补充了 OTTT 和 NDOT 的单独总结文档。
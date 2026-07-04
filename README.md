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
- `Slide/SNNTrainingSlideNotes/`：课程笔记——SNN 训练算法汇总综述（IEEE LaTeX），涵盖 RTRL、E-prop、OTTT、NDOT、S-TLLR、TESS 六个算法的文章汇总与核心思想概述，对应 Week4\_SNN Training.pptx，已编译为 PDF。后续将会和制作PPT一起进行更新。

### Work_Copilot_Drafts/
一系列工作草稿文件，大部分为 vibe_code 内容，包含：
- `Draft.md`：项目初始思路草稿——关于各算法中信息丢失与恢复的核心思想
- `Future_Directions/`：SNN 在线学习与本地学习的未来发展方向与本人研究方向探索（IEEE LaTeX），系统分析信息丢失环节，提出五个未来研究方向与三条具体研究思路
- `Lecture_Notes_Week4/`：Week4_SNN Training.pptx 的课程讲稿（IEEE LaTeX），涵盖 BPTT 回顾、OTTT 核心近似与推导、NDOT 物理重参数化、OTTT vs NDOT 对比分析、实验结果与方法论延伸
- `Algorithm_Summaries/`：OTTT 与 NDOT 算法的单独详细总结（LaTeX + 编译PDF）
- `Meeting_Report/`：会议报告草稿
- `Notes/`：大型笔记文档（`SNNInformationReconstruction.tex`，约 1457 行），涵盖 SNN 信息重建的全面笔记
- `Total_Summary/`：总结与对比分析，包括：
  - `对比分析.md`：OTTT vs NDOT vs S-TLLR vs TESS 四种算法的详细对比（公式对比、精度对比、特性矩阵）
  - `snn_info_summary_cn.tex`：SNN 训练中信息丢失与恢复的中文汇总与研究思路
  - `snn_info_summary.tex`：英文版汇总
  - `SNN中的时间步及其训练机制/`：SNN 时间步相关子专题
---

## Important:

**GroupMeeting&CourseSlide/** 下为组会汇报与课程幻灯片的 LaTeX 源文件及编译产物。主体文件为 `SNN_OnlineLearningSummary.tex`，涵盖**BPTT OTTT NDOT S-TLLR TESS**, 该文档为 LaTeX + IEEEtran 格式，包含讲稿（PPT内容要点 + 口头讲稿），已编译为 PDF。这一部分为重点部分，为主要工作成果结晶，配合组会PPT进行阅读。

---

## 更新日志

### 2026-07-05
- **GroupMeeting&CourseSlide/Slide/SNNTrainingSlideNotes/**：新建 `SNNTrainingSlideNotes.tex`（IEEE 格式），作为课程笔记对 SNN 训练算法进行汇总综述，覆盖 RTRL、E-prop、OTTT、NDOT、S-TLLR、TESS 六篇代表性论文，包含摘要与核心思想概述，对应 Week4\_SNN Training.pptx 课程幻灯片，已编译为 PDF。后续将会和制作PPT一起进行更新。

### 2026-06-30
- **GroupMeeting&CourseSlide/**：完成 `SNN_OnlineLearningSummary.tex`，对 BPTT、OTTT、NDOT、S-TLLR、TESS 五个算法进行了详细论述与数学推导，包含 PPT 内容要点和讲稿。从 BPTT 的标准梯度公式出发，逐步推导至 OTTT 的近似解耦（O(1) 内存在线学习）、NDOT 的物理重参数化（动态 e[t] 替换固定 λ）、S-TLLR 的 STDP 启发三因子规则（时间本地化）、TESS 的 LSG 机制（时空全本地化）。梳理了完整的算法演化路线：BPTT → OTTT → NDOT → S-TLLR → TESS。RTRL 与 E-prop 部分暂缓。
- **Work_Copilot_Drafts/**：维护了大量草稿与 vibe_code 内容，其中 `Total_Summary/对比分析.md` 完成了四种算法的详细横向对比（精度矩阵、特性矩阵、一句话总结），`Total_Summary/snn_info_summary_cn.tex` 汇总了信息丢失与恢复的研究思路，`Notes/SNNInformationReconstruction.tex` 积累了约 1457 行的全面笔记。Algorithm_Summaries/ 下补充了 OTTT 和 NDOT 的单独总结文档。
- **Work_Copilot_Drafts/Future_Directions/**：新建 `SNN_Future_Directions.tex`（IEEE 格式），对 SNN 训练算法演化路径进行系统梳理（BPTT → Online → Local），详细分析了四个信息丢失环节（离散化/时间截断/空间投影近似/资格迹设计），提出了五个未来研究方向（可恢复性理论、自适应时序核、LSG 增强、混合学习策略、多时间尺度学习），并明确了本人聚焦的研究问题——在 O(1) 内存和在线约束下的信息恢复，给出了三条具体探索思路（NDOT 时序核扩展、TESS LSG 的 Hebbian 自适应投影、信息丢失度量与自适应补偿）。
- **Work_Copilot_Drafts/Lecture_Notes_Week4/**：新建 `Week4_SNN_Training_LectureNotes.tex`（IEEE 格式），为 Week4_SNN Training.pptx 编写完整课程讲稿，包含 10 个 Slide 的 PPT 内容要点配口头讲稿：BPTT 瓶颈回顾 → OTTT 近似推导（丢弃重置路径、追踪变量前向递推）→ NDOT 物理重参数化（连续 LIF 方程 → e[t] 闭式解 → 望远镜化简）→ OTTT vs NDOT 核心对比表 → 实验结果 → 方法论延伸（在线到本地的下一站预告）。

---

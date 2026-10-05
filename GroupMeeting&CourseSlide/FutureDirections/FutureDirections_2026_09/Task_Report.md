# 研究任务进度报告

> 本文件汇总三个任务的执行报告，便于后续详细查看。

---

## 任务一：更新研究计划文档 Research_Goal.tex

### 修改的文件（仅 1 个）
`GroupMeeting&CourseSlide\FutureDirections\FutureDirections_2026_09\Research_Goal.tex`

### 实际改动
本轮按用户要求只写了两个 section：

1. **`\section{研究背景与核心问题}`**（标题保留）
   - `\subsection{本地学习缺少全局梯度}`：说明 OTTT / NDOT / S-TLLR / TESS 各自优化局部目标、无贯穿全网的统一梯度；BPTT 有明确的逐层全局误差结构，因此"信息在哪层损失"在本地学习下无答案。
   - `\subsection{核心研究问题}`：点出"看最终准确率"无法回答的三个问题（在哪层损失 / 损失多少 / 为什么损失），落脚到"在层间引入中间量、逐层量化信息保留与训练效果"。

2. **`\section{三个研究方向}`**（新增）
   - `\subsection{方向一：逐层探针}`：探针挂法（时间平均表示 $z^l$ + 线性分类器）→ 探针准确率 $\operatorname{Acc}^l_{\mathrm{probe}}$ → 归一化信息保留率 $\rho^l$ → 曲线判断损失位置。未推互信息下界（按用户要求控制深度）。
   - `\subsection{方向二：逐层表示相似度}`：为什么比较几何结构 → 中心化 Gram 矩阵与相似度 $\operatorname{sim}^l$ 的定义、取值范围 $[-1,1]$、逐层曲线读法。未从 Gram 一路推到 HSIC。
   - `\subsection{方向三：逐层梯度对齐}`：余弦相似度 $\cos^l$ 与平行/垂直分解 $g^l_{\parallel}, g^l_{\perp}$（含 $c^l_{\parallel}, R^l_{\perp}$）的定义与逐层曲线意义。未做更复杂分解。
   - `\subsection{三个方向的关系与整体定位}`：梯度（过程）→ 表示（结果）→ 任务信息（效用）的因果链。

### 保持不变的部分
- preamble、`\input{math_commands_TESS.tex}` / `\input{math_commands_STLLR.tex}`、颜色定义、`\bibliography{SNN_OnlineLearningSummary}` 均未动；
- `\section*{文章汇总}`（六篇文献）原样保留；
- bib key 沿用原 key，未改 `SNN_OnlineLearningSummary.bib`；
- title 与 abstract 更新为新主题（注释同步更新）。

### 复现计划
本轮留空，未写（按用户要求，待后续补充）。

### 编译验证
- `pdflatex` 无 error，输出 `Research_Goal.pdf`（5 页，691352 bytes）；
- `bibtex` 仅 2 条无害 warning（`jiang_ndot_nodate` 的 booktitle/year 为空，属 bib 文件既有问题）；
- 唯一字体 warning（`C70/rm/m/sc` 中文字体形状）为 ctex 常见现象，无害。

---

## 任务二：创建 Project 基础代码

### 创建的文件（11 个）
- `Project/README.md`：项目说明、复现计划、待办清单
- `Project/requirements.txt`：torch、numpy
- `Project/snn/__init__.py`
- `Project/snn/lif.py`：LIF 神经元（仅前向）
- `Project/snn/surrogate.py`：surrogate 梯度函数（fast-sigmoid，为反向预留）
- `Project/network/__init__.py`
- `Project/network/mlp_snn.py`：多层 SNN（仅前向）
- `Project/sanity/__init__.py`
- `Project/sanity/forward_check.py`：手算验证
- `Project/utils/__init__.py`
- `Project/utils/config.py`：全局超参（LAMBDA=0.5, V_TH=1.0）

### forward_check.py 验证结果
在 `SNN_cuda` 环境（Python 3.10.19，torch 2.8.0+cu128，numpy 2.1.2）下运行通过：
- LIF 前向手算（n=2, T=3）膜电位与脉冲均与手算一致 → PASS；
- MLPSNN 输出形状检查 → PASS。

（注：默认 `.venv` 为 Python 3.14，torch 尚不支持，故用 `SNN_cuda` 环境运行。）

### 下一步建议加什么（按依赖顺序）
1. surrogate 反向 + autograd 接入；
2. BPTT-SG（自己实现，作 reference）；
3. OTTT（自己实现，固定衰减 trace）；
4. NDOT（自己实现，动态系数 e^l[t]）；
5. e-prop（必须验证，eligibility/learning-signal 分解）；
6. 逐层评估中间量（探针 / 表示相似度 / 梯度对齐）；
7. 数据集加载与训练循环。

---

## 任务三：修改 PPT 笔记的三个 section

### 修改的三个 section
- BPTT: Full Derivation and Parameter Origins / BPTT 详细推导
- OTTT: From BPTT to Fixed Decay — Full Derivation / OTTT 详细推导
- NDOT: Dynamic Decay — Full Derivation / NDOT 详细推导

### 每个 section 的 Step 数
- BPTT：8 个 Step + 结尾"本节小结与后文关系"
- OTTT：8 个 Step + 结尾"本节小结与前后关系"
- NDOT：8 个 Step + 结尾"本节小结与前后关系"

### 使用的关键符号（沿用原文，未换符号）
- $W^{l\leftarrow l-1}$：权重
- $u^l[t]$、$s^l[t]$：膜电位、脉冲
- $\lambda$、$V_{th}$：衰减因子、阈值
- $\epsilon^l[i]=\lambda I-\lambda V_{th}D_{\Psi,i}$：时间依赖矩阵
- $D_{\Psi,t}$：代理梯度对角矩阵
- $\mathcal{E}_{t,\tau}$：有序时间连乘
- $g_{u^l}[t]$：当前时刻空间梯度
- $\hat{a}^{l-1}[t]$：预突触迹
- $e^l[t]$：NDOT 动态系数

### 是否引入新数学对象 / 新假设
- 未引入任何新数学对象，未引入新假设；
- 主要改动：
  1. 去掉"PPT 内容（英文）"与"公式详细每一步推导"两个子部分的区分，合并为单一详细推导；
  2. 每步统一包含：完整公式 + 解释 + 符号来源与服务作用 + "这一步推导到哪里去"；
  3. 删除所有 `\boxed`（改用普通 equation）；
  4. 删除具体数值例子（OTTT 的 $|x|=1$、$|x|=10$）；
  5. 每个 section 结尾加"本节小结"点明三者关系（BPTT 给出完整 $\epsilon^l[i]$；OTTT 丢弃复位项用固定 $\lambda I$；NDOT 用动态 $e^l[t]$ 替代 $\lambda$）。

### 编译验证
- `pdflatex` 无 error，输出 `SNNTrainingSlideNotes.pdf`（29 页，929712 bytes）；
- 唯一字体 warning（`C70/rm/m/sc`）为 ctex 常见现象，无害。

---

## 最终总结（三个任务总览）

### 任务一：更新研究计划文档 Research_Goal.tex
- 只写了两个 section：研究背景与核心问题 + 三个研究方向（逐层探针、逐层表示相似度、逐层梯度对齐），推导深度按用户要求控制；
- 复现计划留空（待后续补充）；preamble、bib、"文章汇总"、section 标题均保留；
- 编译通过（5 页 PDF，无 error）。

### 任务二：创建 Project 基础代码（11 个文件）
- `README.md`、`requirements.txt`（torch/numpy）、`snn/lif.py`（仅前向）、`snn/surrogate.py`、`network/mlp_snn.py`（多层前向）、`sanity/forward_check.py`、`utils/config.py` 及 4 个 `__init__.py`；
- forward_check.py 跑通：n=2、T=3 手算对照 PASS + MLPSNN 形状检查 PASS（在 `SNN_cuda` 环境运行，torch 2.8.0）；
- 未建 BPTT/OTTT/NDOT/e-prop、探针、数据集等（按依赖顺序列在 README 待办里）。

### 任务三：修改 PPT 笔记三个 section
- 重写 BPTT、OTTT、NDOT 三个 section，各 8 个 Step + 结尾"本节小结"；
- 去掉英文 PPT 子部分与 `\boxed`、删除数值例子，统一为"公式 + 解释 + 符号来源 + 推导方向"的详细推导；
- 符号沿用原文（`W^{l←l-1}`、`ε^l[i]`、`D_{Ψ,t}`、`â^{l-1}[t]`、`e^l[t]`），未引入新对象/新假设；
- 编译通过（29 页 PDF，无 error）。

### 说明
- 任务报告已按用户要求统一写在 `Work_Copilot_Drafts\Task_Report.md`；
- 编译/运行都用了 `SNN_cuda` 环境（默认 `.venv` 是 Python 3.14，torch 尚不支持）；
- 唯一反复出现的是 ctex 中文字体 shape 的无害 warning，与内容无关。

### 可继续推进的方向
- 任务一：补充"复现计划"与"战略目标"（此前按用户要求留空）；
- 任务二：下一批代码（surrogate 反向 + autograd 接入 → BPTT-SG）。

---


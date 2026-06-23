# S-TLLR：受 STDP 启发的脉冲神经网络时间局部学习规则（中文完整整理）

## 1. 论文基本信息

- **论文标题**：S-TLLR: STDP-inspired Temporal Local Learning Rule for Spiking Neural Networks
- **中文标题**：S-TLLR：受 STDP 启发的脉冲神经网络时间局部学习规则
- **作者**：Marco P. E. Apolinario、Kaushik Roy
- **单位**：Purdue University, School of Electrical and Computer Engineering
- **arXiv 版本**：arXiv:2306.15220v4，2024 年 10 月 29 日
- **代码**：论文说明代码位于 GitHub repository

---

## 2. 摘要核心内容

论文研究的是如何在脉冲神经网络（Spiking Neural Networks, SNNs）中实现高效训练，尤其是在事件驱动任务和边缘设备场景下实现低内存、低计算开销的在线学习。

SNN 具有生物合理性，并且适合在边缘设备上部署低能耗智能系统，特别适合处理序列学习任务。但是，SNN 的训练面临两个核心困难：

1. **时间信用分配（temporal credit assignment）**：需要判断某个早期脉冲对之后输出和损失的影响。
2. **空间信用分配（spatial credit assignment）**：需要判断不同层、不同神经元对最终误差的贡献。

传统 BPTT 能够解决这两个问题，但需要沿时间展开网络并保存所有时间步的状态，因此计算和显存开销高，且不适合边缘设备在线学习。

论文提出 **S-TLLR（STDP-inspired Temporal Local Learning Rule）**，这是一种受 Spike-Timing Dependent Plasticity（STDP）启发的三因子时间局部学习规则。S-TLLR 用于训练深层 SNN，特别是事件驱动学习任务。它具有以下特点：

- 使用三因子学习形式；
- 使用受 STDP 启发的资格迹（eligibility trace）；
- 同时利用 causal 与 non-causal 的脉冲时序关系；
- 显存复杂度与时间步数无关；
- 时间复杂度也可以不随完整 BPTT 时间链增长；
- 适合低功耗边缘设备上的在线学习。

论文在图像识别、手势识别、音频分类、事件相机光流估计等任务上验证了该方法。实验显示，S-TLLR 能取得与 BPTT 相近的准确率，同时显存减少约 **5 到 50 倍**，MAC 操作减少约 **1.3 到 6.6 倍**。

---

## 3. 引言

### 3.1 背景动机

过去十多年，人工智能的发展很大程度上依赖模型规模和复杂度的持续增加。大型模型在认知任务上取得了显著进展，但也带来了很高的能耗和计算资源需求。

相比之下，人脑具有极高的能效。因此，研究者对类脑计算产生了浓厚兴趣，希望模仿生物神经元的一些关键特征，例如：

- spike-based communication：基于脉冲的通信；
- sparsity：稀疏活动；
- spatio-temporal processing：时空信息处理。

SNN 正是这种方向上的代表模型。它使用二值脉冲事件进行通信，在合适硬件上可以显著降低推理能耗。

### 3.2 SNN 的优势

SNN 的主要优势包括：

- 事件驱动计算；
- 二值稀疏脉冲；
- 膜电位积分产生天然的时间处理能力；
- 适合神经形态硬件；
- 适合边缘设备中的序列学习任务。

### 3.3 SNN 训练困难

尽管 SNN 有推理优势，但训练 SNN 很难。原因是训练需要解决：

- 时间信用分配问题；
- 空间信用分配问题。

BPTT 是最常用的方法，但它需要沿时间展开网络。若一个层有 $n$ 个神经元、输入序列有 $T$ 个时间步，则 BPTT 的典型开销为：

- 内存复杂度：$O(Tn)$；
- 时间复杂度：$O(Tn^2)$。

这使得 BPTT 不适合受限于内存和能耗的边缘设备。

### 3.4 现有时间局部方法的问题

已有一些方法尝试以常数内存近似 BPTT，例如：

- RTRL；
- e-prop；
- OSTL；
- ETLP；
- OSTTP；
- OTTT。

但是，大多数方法的复杂度随着突触数增长，即 $O(n^2)$，在深度卷积 SNN 中仍然很昂贵。并且，这些方法大多来自 BPTT 近似，因此主要利用 pre-synaptic 与 post-synaptic 活动之间的 **causal relation**，而忽略 STDP 机制中存在的 **non-causal relation**。

---

## 4. 图 1 的含义

论文图 1 比较了三种权重更新方式：

1. **BPTT**
2. **STDP**
3. **S-TLLR**

图中：

- 绿色信号表示自上而下的学习信号；
- 红色信号表示层内局部可得的 causal term；
- 蓝色信号表示 non-causal term。

关键区别是：

- BPTT 的学习信号依赖未来时间步，因此不是时间局部的；
- STDP 使用局部脉冲时序关系，但缺少任务误差信号；
- S-TLLR 使用 STDP 风格的 causal 与 non-causal 资格迹，同时用学习信号调制，从而形成三因子学习规则。

---

## 5. 表 1：复杂度与局部性比较

论文比较了多种 SNN 学习方法。设 $n$ 为神经元数量，$T$ 为时间步数。

| 方法 | 内存复杂度 | 时间复杂度 | 时间局部 | 利用 non-causality |
|---|---|---|---|---|
| BPTT | $O(Tn)$ | $O(Tn^2)$ | 否 | 否 |
| RTRL | $O(n^3)$ | $O(n^4)$ | 是 | 否 |
| e-prop | $O(n^2)$ | $O(n^2)$ | 是 | 否 |
| OSTL | $O(n^2)$ | $O(n^2)$ | 是 | 否 |
| ETLP | $O(n^2)$ | $O(n^2)$ | 是 | 否 |
| OSTTP | $O(n^2)$ | $O(n^2)$ | 是 | 否 |
| OTTT | $O(n)$ | $O(n^2)$ | 是 | 否 |
| S-TLLR | $O(n)$ | $O(n^2)$ | 是 | 是 |

S-TLLR 的主要特点是：

- 与 OTTT 一样具有线性内存复杂度；
- 与其他 BPTT 近似方法不同，它利用 non-causal spike-timing relation；
- 适合在线训练。

---

## 6. 论文贡献

论文贡献可归纳为四点：

1. **提出 S-TLLR**：一种受 STDP 启发的时间局部学习规则，用于训练 SNN，内存复杂度与神经元数线性相关，并且不随时间步数增加。
2. **证明 non-causal 关系有益**：通过实验展示，在 SNN 学习中引入 non-causal spike timing 可以改善泛化能力和任务性能。
3. **验证多种网络结构**：S-TLLR 可用于 VGG、ResNet、U-Net-like、recurrent 等架构。
4. **拓展事件驱动应用**：在图像识别、手势识别、音频分类、事件相机光流估计等多种任务上验证有效性。

---

# 7. 背景知识

## 7.1 SNN 与 LIF 神经元

论文使用 Leaky Integrate-and-Fire（LIF）模型来描述脉冲神经元动力学：

$$
u_i[t] = \gamma (u_i[t-1] - v_{th} y_i[t-1]) + w_{ij} x_j[t]
$$

$$
y_i[t] = \Theta (u_i[t] - v_{th})
$$

其中：

- $u_i[t]$：第 $i$ 个神经元在时间 $t$ 的膜电位；
- $w_{ij}$：从第 $j$ 个突触前神经元到第 $i$ 个突触后神经元的权重；
- $x_j[t]$：突触前神经元输入脉冲；
- $y_i[t]$：突触后神经元输出脉冲；
- $\gamma$：泄漏因子；
- $v_{th}$：阈值；
- $\Theta$：Heaviside 函数。

当膜电位达到阈值时，神经元产生二值脉冲。脉冲产生后，通过 $v_{th} y_i[t]$ 实现 subtractive reset。

---

## 7.2 STDP

STDP 是生物神经系统中观察到的一种突触可塑性机制。它描述突触强度如何根据突触前和突触后神经元发放脉冲的时间顺序变化。

传统 STDP 的基本规则：

- 若突触前神经元在突触后神经元之前发放，则突触增强；
- 若突触前神经元在突触后神经元之后发放，则突触减弱。

即：STDP 奖励因果性，惩罚非因果性。

但论文引用 Anisimova 等人的观点，指出 STDP 中偏好因果性可能只是暂时现象，长期来看 STDP 可能倾向于奖励同步性，即同时利用 causal 和 non-causal relation。

STDP 的一般形式为：

$$
\Phi(t_i, t_j) =
\begin{cases}
\alpha_{\text{pre}} \lambda_{\text{pre}}^{t_i - t_j}, & \text{if } t_i \geq t_j \quad \text{(causal term)} \\[4pt]
\alpha_{\text{post}} \lambda_{\text{post}}^{t_j - t_i}, & \text{if } t_i < t_j \quad \text{(non-causal term)}
\end{cases}
$$

其中：

- $t_i$：突触后神经元发放时间；
- $t_j$：突触前神经元发放时间；
- $\alpha_{\text{pre}}$：causal term 强度；
- $\lambda_{\text{pre}}$：causal term 衰减因子；
- $\alpha_{\text{post}}$：non-causal term 强度；
- $\lambda_{\text{post}}$：non-causal term 衰减因子。

当 $\alpha_{\text{post}} < 0$ 时，STDP 更偏向 causality；当 $\alpha_{\text{post}} > 0$ 时，STDP 更偏向 synchrony。

---

## 7.3 STDP 的前向递推形式

STDP 的权重变化可写为：

$$
\begin{aligned}
\Delta w_{ij}[t] &= \alpha_{\text{pre}} y_i[t] \sum_{t'=0}^{t} \lambda_{\text{pre}}^{t-t'} x_j[t'] \\
&\quad + \alpha_{\text{post}} x_j[t] \sum_{t'=0}^{t-1} \lambda_{\text{post}}^{t-t'} y_i[t']
\end{aligned}
$$

第一项是 causal term：当前 post-synaptic 活动与过去 pre-synaptic 活动相关。

第二项是 non-causal term：当前 pre-synaptic 活动与过去 post-synaptic 活动相关。

两项都可以通过 trace 前向递推计算：

$$
\mathrm{tr}(x_j)[t] = \lambda_{\text{pre}} \mathrm{tr}(x_j)[t-1] + x_j[t]
$$

因此 STDP 更新可写成：

$$
\Delta w_{ij}[t] = \alpha_{\text{pre}} y_i[t] \, \mathrm{tr}(x_j)[t] + \alpha_{\text{post}} x_j[t] \left( \mathrm{tr}(y_i[t]) - y_i[t] \right)
$$

---

## 7.4 BPTT 与三因子学习规则

BPTT 是默认训练 SNN 的梯度算法。它通过展开所有时间步并应用链式法则计算梯度：

$$
\frac{d\mathcal{L}}{dw} = \sum_t \frac{\partial\mathcal{L}}{\partial y[t]} \frac{\partial y[t]}{\partial u[t]} \frac{\partial u[t]}{\partial w}
$$

BPTT 能有效解决时间和空间信用分配问题，但计算需求随时间步增长，并且缺乏生物合理性。

三因子学习规则（three-factor learning rule）更接近生物可塑性。它使用三类信号：

1. 突触前活动；
2. 突触后活动；
3. 自上而下的学习信号。

三因子规则中，只有当资格迹 $e_{ij}$ 存在时，突触才会被更新。一般形式为：

$$
e_{ij}[t] = \beta e_{ij}[t-1] + f(y_i[t]) \, g(x_j[t])
$$

权重更新为：

$$
\Delta w_{ij} = \sum_t \delta_i[t] \, e_{ij}[t]
$$

其中：

- $e_{ij}[t]$：资格迹；
- $\delta_i[t]$：学习信号；
- $f(y_i[t])$：突触后活动函数；
- $g(x_j[t])$：突触前活动函数；
- $\beta$：资格迹衰减因子。

许多三因子学习规则可以近似 BPTT，但若为每个突触保存资格迹，内存复杂度为 $O(n^2)$，对深度卷积 SNN 很昂贵。

---

# 8. 相关工作

## 8.1 Surrogate Gradient 与 BPTT

Surrogate gradient 方法用连续函数近似不可导的脉冲发放函数，使误差能够反向传播。它可以利用单个 spike 的时间信息，适合图像分类之外的多种任务，也能产生低延迟模型。

缺点是：

- 必须结合 BPTT；
- 显存和计算开销随时间步数线性增加；
- 不适合边缘设备在线训练。

## 8.2 生物启发学习规则

STDP 不需要外部监督信号，因此适合片上学习，但传统 STDP 有明显限制：

- 需要大量训练样本；
- 准确率通常较低；
- 难以训练深层网络；
- 难以解决复杂机器学习任务。

三因子学习规则通过资格迹和误差信号改善了 STDP 的限制，但很多方法仍有 $O(n^2)$ 的时间或空间复杂度。

## 8.3 处理时间依赖问题的方法

为避免 BPTT 的时间展开，已有方法包括：

- RTRL：可计算精确梯度，但复杂度极高；
- e-prop、OSTL、OTTT：利用时间局部信息；
- ETLP、OSTTP：三因子局部学习方法。

这些方法大多只使用 causal relation，而 S-TLLR 的特色是进一步利用 non-causal relation。

## 8.4 STDP 与反向传播结合

已有方法尝试把 STDP 与反向传播结合，例如：

- 先用 STDP 预训练，再用 BPTT 微调；
- 用误差信号调制 STDP。

但这些方法要么不能解决 BPTT 的时间依赖问题，要么无法扩展到深层 SNN 或复杂视觉任务。

---

# 9. S-TLLR 方法

## 9.1 方法概述

S-TLLR 是一种新的三因子学习规则，受 STDP 启发。它的核心特征是：

- 时间局部；
- 利用 causal 与 non-causal spike-timing relation；
- 低内存复杂度 $O(n)$；
- 可以结合 BP 或 DFA 生成学习信号。

常规三因子规则需要保存每个突触的资格迹 $e_{ij}$，内存复杂度为 $O(n^2)$。S-TLLR 通过去掉资格迹中的 recurrent term，即令 $\beta = 0$，只保留 instantaneous term，从而不必为每个突触保存状态。

S-TLLR 只需保存两个神经元级变量：

- pre-synaptic trace；
- post-synaptic trace。

因此内存复杂度从 $O(n^2)$ 降为 $O(n)$。

---

## 9.2 S-TLLR 的三因子形式

S-TLLR 的权重更新形式为：

$$
\Delta w_{ij}[t] = \delta_i[t] \, e_{ij}[t]
$$

其中：

- $\delta_i[t]$：自上而下的学习信号；
- $e_{ij}[t]$：基于 STDP 的即时资格迹。

---

## 9.3 广义 STDP 资格迹

S-TLLR 使用广义 STDP 方程来计算资格迹：

$$
\begin{aligned}
e_{ij}[t] &= \alpha_{\text{pre}} \Psi(u_i[t]) \sum_{t'=0}^{t} \lambda_{\text{pre}}^{t-t'} x_j[t'] \\
&\quad + \alpha_{\text{post}} x_j[t] \sum_{t'=0}^{t-1} \lambda_{\text{post}}^{t-t'} \Psi(u_i[t'])
\end{aligned}
$$

其中：

- $\Psi$：secondary activation function；
- $u_i[t]$：突触后神经元膜电位；
- $x_j[t]$：突触前脉冲；
- $\alpha_{\text{pre}}$：causal term 强度；
- $\lambda_{\text{pre}}$：causal trace 衰减；
- $\alpha_{\text{post}}$：non-causal term 强度；
- $\lambda_{\text{post}}$：non-causal trace 衰减。

$\Psi$ 可以不同于发放函数 $\Theta$。论文经验上发现，使用满足 $\int \Psi(u) du \leq 1$ 的函数效果更好。

---

## 9.4 前向递推实现

上述资格迹可以写为前向时间递推：

$$
\begin{aligned}
e_{ij}[t] &= \alpha_{\text{pre}} \Psi(u_i[t]) \, \mathrm{tr}(x_j)[t] \\
&\quad + \alpha_{\text{post}} y_i[t] \left( \mathrm{tr}(\Psi(u_i[t])) - \Psi(u_i[t]) \right)
\end{aligned}
$$

第一项：

- 代表 causal relation；
- 当前 post-synaptic 活动与过去 pre-synaptic 活动相关；
- 用红色信号表示。

第二项：

- 代表 non-causal relation；
- 当前 pre-synaptic 活动与过去 post-synaptic 活动相关；
- 用蓝色信号表示。

S-TLLR 的重要点是：这两个 trace 都可以 forward-in-time 计算，不需要反向穿越时间。

---

## 9.5 学习信号

学习信号定义为：

$$
\delta_i^{(l)}[t] =
\begin{cases}
\dfrac{\partial \mathcal{L}(y^L[t], y^*)}{\partial y_i^{(l)}[t]}, & \text{if } t \geq T_l \\[6pt]
0, & \text{otherwise}
\end{cases}
$$

其中：

- $T_l$ 是开始提供学习信号的时间步；
- $y^*$ 是真实标签；
- $y^L[t]$ 是输出层在时间 $t$ 的输出；
- $\mathcal{L}$ 是损失函数。

权重更新为：

$$
w_{ij} := w_{ij} + \rho \sum_{t=T_l}^{T} \delta_i[t] \, e_{ij}[t]
$$

论文强调：S-TLLR 中的反向传播只在层之间进行，而不沿时间传播，因此是 **temporally local** 的。根据任务不同，即使只在最后一个时间步提供学习信号，也可以得到较好性能。

---

## 9.6 使用 DFA 的可能性

论文主要使用 error-backpropagation 生成学习信号。但也指出可以使用随机反馈连接，例如 Direct Feedback Alignment（DFA），从输出层直接向隐藏层提供固定随机反馈。

若使用 DFA，S-TLLR 也可以具备空间局部性。但论文主实验重点仍然是 BP 生成学习信号的情形。

---

## 9.7 具有循环连接的模型

对于显式循环连接，LIF 模型写为：

$$
u_i[t] = \gamma(u_i[t-1] - v_{th} y_i[t-1]) + w_{ij}^{\text{ff}} x_j[t] + w_{ik}^{\text{rec}} y_k[t-1]
$$

$$
y_i[t] = \Theta(u_i[t] - v_{th})
$$

其中：

- $w_{ij}^{\text{ff}}$ 是前馈连接；
- $w_{ik}^{\text{rec}}$ 是同层循环连接。

循环连接的资格迹改为：

$$
\begin{aligned}
e_{ik}^{\text{rec}}[t] &= \alpha_{\text{pre}} \Psi(u_i[t]) \sum_{t'=1}^{t} \lambda_{\text{pre}}^{t-t'} y_k[t'-1] \\
&\quad + \alpha_{\text{post}} y_k[t-1] \sum_{t'=0}^{t-1} \lambda_{\text{post}}^{t-t'} \Psi(u_i[t'])
\end{aligned}
$$

该形式仍保持常数内存和时间局部性。

---

# 10. 计算与内存分析

## 10.1 BPTT 的开销

对于 $L$ 层、每层神经元数为 $N^{(l)}$ 的网络，BPTT 需要保存所有时间步的状态：

$$
\text{Mem}_{\text{BPTT}} = T \times \sum_l N^{(l)}
$$

BPTT 的 MAC 操作估计为：

$$
\text{MAC}_{\text{BPTT}} = 2T \times \sum_l N^{(l)} \times N^{(l-1)}
$$

其中 factor 2 来自：

1. 权重梯度计算；
2. 学习信号向前一层传播。

## 10.2 S-TLLR 的开销

S-TLLR 的内存为：

$$
\text{Mem}_{\text{S-TLLR}} = 2 \times \sum_l N^{(l)}
$$

factor 2 来自 causal 与 non-causal trace 变量。

S-TLLR 的 MAC 操作为：

$$
\text{MAC}_{\text{S-TLLR}} = 3(T - T_l) \times \sum_l N^{(l)} \times N^{(l-1)}
$$

factor 3 来自：

- causal term；
- non-causal term；
- 学习信号调制权重更新。

## 10.3 改进比例

显存降低比例：

$$
S_{\text{mem}} = \frac{\text{Mem}_{\text{BPTT}}}{\text{Mem}_{\text{S-TLLR}}} = \frac{T}{2}
$$

MAC 降低比例：

$$
S_{\text{MAC}} = \frac{\text{MAC}_{\text{BPTT}}}{\text{MAC}_{\text{S-TLLR}}} = \frac{2T}{3(T - T_l)}
$$

当学习信号只在最后几个时间步提供时，$T - T_l$ 很小，因此 MAC 降低更明显。

---

# 11. S-TLLR 算法流程

算法输入：

- 输入 $x$；
- 标签 $y^*$；
- 权重 $w$；
- 总时间步 $T$；
- 学习信号起始时间 $T_l$；
- 学习率 $\rho$。

流程：

1. 对每个时间步 $t = 1, \dots, T$：
   - 对每层更新膜电位；
   - 产生输出脉冲；
   - 更新资格迹。
2. 若 $t \geq T_l$：
   - 初始化输出层学习信号；
   - 从输出层向隐藏层生成学习信号，可用 BP 或 DFA；
   - 输出层权重更新使用输出误差信号与上一层脉冲；
   - 隐藏层权重更新使用 $\delta_i^{(l)} e_{ij}^{(l-1)}$。
3. 对每层累积更新权重。

---

# 12. 实验评估

论文实验覆盖多种事件驱动任务：

- DVS Gesture：手势识别；
- DVS CIFAR10：事件图像识别；
- N-CALTECH101：事件图像识别；
- SHD：音频分类；
- MVSEC：事件相机光流估计。

---

## 12.1 non-causal term 消融实验

论文在 DVS Gesture、DVS CIFAR10、N-CALTECH101、SHD 上研究 $\alpha_{\text{post}}$ 的影响。

$\alpha_{\text{post}}$ 取值：

- $\alpha_{\text{post}} = 0$：只使用 causal term；
- $\alpha_{\text{post}} = +1$：正向加入 non-causal term；
- $\alpha_{\text{post}} = -1$：负向加入 non-causal term。

结果：

| 数据集 | 模型 | $T$ | $T_l$ | $\alpha_{\text{post}}=0$ | $\alpha_{\text{post}}=+1$ | $\alpha_{\text{post}}=-1$ |
|---|---|---|---|---|---|---|
| DVS Gesture | VGG9 | 20 | 15 | 94.61±0.73% | 94.01±1.10% | 95.07±0.48% |
| DVS CIFAR10 | VGG9 | 10 | 5 | 72.93±0.94% | 73.42±0.50% | 73.93±0.62% |
| N-CALTECH101 | VGG9 | 10 | 5 | 62.24±1.22% | 53.42±1.50% | 66.33±0.86% |
| SHD | RSNN | 100 | 10 | 77.09±0.33% | 78.23±1.84% | 74.69±0.47% |

结论：

- 对视觉任务，$\alpha_{\text{post}} = -1$ 通常更好；
- 对 SHD 音频任务，$\alpha_{\text{post}} = +1$ 更好；
- 引入 non-causal relation 通常有助于性能；
- non-causal term 可能起到正则化作用，使权重空间探索更充分。

---

## 12.2 图像和手势识别

实验设置：

- VGG-9 与 ResNet18；
- Adam 优化器；
- 学习率 0.001；
- 训练 300 epochs；
- 5 个随机种子；
- DVS Gesture 使用 $(\lambda_{\text{post}}, \lambda_{\text{pre}}, \alpha_{\text{post}}, \alpha_{\text{pre}}) = (0.2, 0.75, -1, 1)$；
- DVS CIFAR10 与 N-CALTECH101 使用 $(0.2, 0.5, -1, 1)$。

### DVS CIFAR10

| 方法 | 模型 | $T$ | Batch | 准确率 | MAC | Memory |
|---|---|---|---|---|---|---|
| BPTT baseline | VGG-9 | 10 | 48 | 75.44±0.76% | 6.82×10⁹ | 18.12 MB |
| S-TLLR, $T_l$=5, $\alpha_{\text{post}}$=-1 | VGG-9 | 10 | 48 | 73.93±0.62% | 5.12×10⁹ | 3.62 MB |
| S-TLLR, $T_l$=0, $\alpha_{\text{post}}$=-1 | VGG-9 | 10 | 48 | 75.6±0.10% | 10.26×10⁹ | 3.62 MB |
| S-TLLR, $T_l$=0, $\alpha_{\text{post}}$=0 | VGG-9 | 10 | 48 | 74.8±0.15% | 6.82×10⁹ | 3.62 MB |
| BPTT baseline | ResNet18 | 10 | 48 | 72.68±0.87% | 7.13×10⁹ | 28.14 MB |
| S-TLLR, $T_l$=5 | ResNet18 | 10 | 48 | 71.94±0.75% | 5.12×10⁹ | 5.62 MB |
| S-TLLR, $T_l$=0 | ResNet18 | 10 | 48 | 74.5±0.64% | 10.24×10⁹ | 5.62 MB |

结论：

- S-TLLR 在 $T_l=0$ 时可达到或超过 BPTT baseline；
- S-TLLR 显存显著低于 BPTT；
- non-causal term 带来性能提升。

### DVS Gesture

| 方法 | 模型 | $T$ | Batch | 准确率 | MAC | Memory |
|---|---|---|---|---|---|---|
| BPTT baseline | VGG-9 | 20 | 16 | 95.58±1.08% | 6.06×10⁹ | 16.13 MB |
| S-TLLR | VGG-9 | 20 | 16 | 97.72±0.38% | 2.27×10⁹ | 1.61 MB |
| BPTT baseline | ResNet18 | 20 | 16 | 94.92±0.38% | 6.34×10⁹ | 25.03 MB |
| S-TLLR | ResNet18 | 20 | 16 | 94.92±0.61% | 2.27×10⁹ | 2.50 MB |

结论：

- 在 DVS Gesture 上，S-TLLR 的 VGG-9 结果优于 BPTT；
- 论文认为 BPTT 在小数据集上容易过拟合，而 S-TLLR 的简单形式和只在最后几个时间步更新有助于缓解过拟合；
- 显存约减少 10 倍。

### N-CALTECH101

| 方法 | 模型 | $T$ | Batch | 准确率 | MAC | Memory |
|---|---|---|---|---|---|---|
| BPTT baseline | VGG-9 | 10 | 16 | 65.92±0.82% | 22.81×10⁹ | 20.15 MB |
| S-TLLR | VGG-9 | 10 | 16 | 66.058±0.92% | 17.22×10⁹ | 4.03 MB |
| BPTT baseline | ResNet18 | 10 | 16 | 60.89±0.89% | 6.34×10⁹ | 31.31 MB |
| S-TLLR | ResNet18 | 10 | 16 | 61.65±0.99% | 4.27×10⁹ | 6.26 MB |

结论：S-TLLR 在 N-CALTECH101 上也取得与 BPTT 相当甚至略优的结果，同时显存显著降低。

---

## 12.3 音频分类 SHD

模型：LIF-RSNN。

| 方法 | 模型 | $T$ | Batch | 准确率 | MAC | Memory |
|---|---|---|---|---|---|---|
| BPTT baseline | LIF-RSNN | 100 | 128 | 70.57±0.96% | 0.054×10⁹ | 0.961 MB |
| S-TLLR_BP | LIF-RSNN | 100 | 128 | 78.24±1.84% | 0.096×10⁹ | 0.019 MB |
| S-TLLR_DFA | LIF-RSNN | 100 | 128 | 74.60±0.52% | 0.096×10⁹ | 0.019 MB |

结论：

- S-TLLR_BP 超过 BPTT baseline；
- S-TLLR_DFA 也具有竞争力；
- S-TLLR 的显存减少约 50 倍；
- S-TLLR 有助于避免 BPTT 在 RSNN 上的过拟合。

---

## 12.4 事件相机光流估计

论文使用 MVSEC 数据集，评价指标为 Average Endpoint Error（AEE），越低越好：

$$
\text{AEE} = \frac{1}{P} \sum_P \| y_{\text{pred}}(i,j) - y_{\text{gt}}(i,j) \|_2
$$

实验模型：Fully-Spiking FlowNet（FSFN）。

训练设置：

- Adam；
- 学习率 0.0002；
- batch size 8；
- 100 epochs；
- $T_l = 9$，即只在最后时间步使用学习信号；
- 参数 $(\lambda_{\text{post}}, \lambda_{\text{pre}}, \alpha_{\text{pre}}) = (0.5, 0.8, 1)$；
- $\alpha_{\text{post}} \in \{-0.2, 0.2, 0\}$。

结果：

| 模型 | 方法 | OD1 | IF1 | IF2 | IF3 | AEE Sum |
|---|---|---|---|---|---|---|
| FSFN $\alpha_{\text{post}}=-0.2$ | S-TLLR | 0.50 | 0.76 | 1.19 | 1.00 | 3.45 |
| FSFN $\alpha_{\text{post}}=0.2$ | S-TLLR | 0.54 | 0.78 | 1.28 | 1.09 | 3.69 |
| FSFN $\alpha_{\text{post}}=0$ | S-TLLR | 0.50 | 0.77 | 1.25 | 1.08 | 3.60 |
| FSFN baseline | BPTT | 0.45 | 0.76 | 1.17 | 1.02 | 3.40 |

结论：

- S-TLLR 在复杂时空任务光流估计上也接近 BPTT；
- $\alpha_{\text{post}}=-0.2$ 的 S-TLLR 是 S-TLLR 设置中最好；
- 与 BPTT 相比，内存约减少 5 倍，MAC 减少 6.6 倍。

---

# 13. 附录内容

## 13.1 数据集与实验设置

论文使用的数据集包括：

- DVS Gesture；
- N-CALTECH101；
- DVS CIFAR10；
- SHD；
- MVSEC。

这些数据集覆盖：

- 图像识别；
- 手势识别；
- 音频分类；
- 光流估计。

## 13.2 网络结构

### VGG-9

用于图像和手势识别：

```
64C3-128C3-AP2S2-256C3-256C3-AP2S2-512C3-512C3-AP2S2-512C3-512C3-AP2S2-FC
```

其中：

- `64C3` 表示输出通道 64、卷积核 3×3；
- `AP2S2` 表示 2×2 average pooling，stride=2；
- `FC` 表示全连接层。

该模型不用 batch normalization，而采用 weight standardization。LIF 参数为：

$$
\gamma = 0.5, \quad v_{th} = 0.8
$$

### SHD 的 RSNN

- 一个 450 神经元的 recurrent layer；
- 一个 20 神经元的 leaky integrator readout layer；
- 两层 leak factor 都为 $\gamma = 0.99$；
- recurrent LIF threshold 为 $v_{th} = 0.8$。

### FSFN

用于光流估计，采用 U-Net-like 架构，所有层使用二值 spike 计算，并在卷积层加入 weight standardization。

FSFN 参数：

$$
T = 10, \quad \gamma = 0.88, \quad v_{th} = 0.6
$$

---

## 13.3 数据预处理

### DVS Gesture

- 11 类手势；
- 29 个主体；
- 3 种照明条件；
- DVS 分辨率 128×128；
- 切分为 1.5 秒序列；
- 累积为 20 个 event frames，每个 75 ms；
- resize 到 32×32；
- 正负极性作为通道。

### N-CALTECH101

- CALTECH101 的事件版本；
- 使用 ATIS 事件相机记录显示器上静态图像；
- 分为 10 个时间 bin，每个 30 ms；
- resize 到 60×45。

### DVS CIFAR10

- CIFAR10 的事件相机版本；
- 10,000 个样本；
- 原始分辨率 128×128；
- 分为 10 个时间 bin；
- resize 到 48×48；
- 使用 padding=4 的 random crop 数据增强。

### SHD

- 音频分类数据集；
- 包含英语和德语数字 0 到 9；
- 音频转为 spike trains；
- 分为 100 个 bin，每个 10 ms；
- 不使用数据增强。

### MVSEC

- 用于事件相机光流预测；
- 包含 indoor flying 和 outdoor driving；
- 提供 ground truth optical flow；
- 两帧灰度图之间的事件被分为 10 个 bin；
- 保留正负极性通道。

---

## 13.4 损失函数与 secondary activation function

分类任务使用交叉熵损失。

光流任务使用基于 photometric loss 与 smooth loss 的自监督损失。

学习信号生成时间：

- 图像和手势识别：最后 5 个时间步；
- 音频分类：$T_l = 90$；
- 光流估计：$T_l = 1$；

这分别减少计算量约 4×、1.1×、10×。

论文考虑了多种 secondary activation function $\Psi$：

$$
\Psi(u_i[t]) = \frac{1}{(100 |u_i[t] - v_{th}| + 1)^2}
$$

$$
\Psi(u_i[t]) = 0.3 \times \max(1.0 - |u_i[t] - v_{th}|, 0)
$$

$$
\Psi(u_i[t]) = 4 \times \sigma(u_i[t] - v_{th}) (1 - \sigma(u_i[t] - v_{th}))
$$

$$
\Psi(u_i[t]) = \frac{1}{1 + (10(u_i[t] - v_{th}))^2}
$$

这些函数类似 BPTT 中 surrogate gradient 的作用。

## 13.5 BPTT 分析

附录说明，BPTT 的误差信号在时间 $t'$ 依赖未来时间步 $t'+1, \dots, T$，因此不能在当前时间步局部计算。

BPTT 的权重更新可分为两部分：

1. 当前和过去可得的资格迹样成分；
2. 依赖未来时间的学习信号。

正是第二部分导致 BPTT 不是时间局部的。

## 13.6 实际 GPU 显存

附录使用一个五层全连接 SNN 进行简单回归实验，时间步取 10、25、50、100、200、300。结果显示：

- BPTT 显存随时间步数线性增长；
- S-TLLR 显存基本保持常数。

## 13.7 其他消融

### 使用 DFA 生成学习信号

使用 DFA 时，加入 non-causal term 与否差别不大。论文认为，这说明需要更精确的学习信号，如 BP，才能充分发挥 non-causal term 的优势。

### $\lambda_{\text{pre}}$ 的影响

论文发现，$\lambda_{\text{pre}}$ 不一定要等于 LIF 泄漏参数 $\gamma$。实验表明，略高的 $\lambda_{\text{pre}}$ 可能带来更好的平均准确率。

### secondary activation function 的影响

附录表 5 表明，在不同 $\Psi$ 函数下，使用非零 $\alpha_{\text{post}}$ 往往优于 $\alpha_{\text{post}}=0$，进一步支持 non-causal term 有益这一结论。

---

# 14. 结论

S-TLLR 是一种受 STDP 启发的三因子时间局部学习规则。与 BPTT 相比，它不需要沿时间反向传播，显存需求不随时间步增长。与许多三因子学习规则相比，它避免了为每个突触保存递归资格迹，从而把内存复杂度降低到 $O(n)$。

S-TLLR 的主要创新在于：

1. 使用广义 STDP 资格迹；
2. 同时利用 causal 和 non-causal spike-timing relation；
3. 通过学习信号调制资格迹形成三因子学习；
4. 保持时间局部和低内存复杂度；
5. 可用于多种事件驱动任务。

实验显示：

- S-TLLR 在 DVS Gesture、DVS CIFAR10、N-CALTECH101、SHD、MVSEC 等任务上能达到与 BPTT 相当的性能；
- 显存减少约 5 到 50 倍；
- MAC 操作减少约 1.3 到 6.6 倍；
- 在视觉任务中，负向 non-causal term 往往更好；
- 在时间信息主导的 SHD 音频任务中，同步性即正向 non-causal term 更有益。

总体而言，S-TLLR 为低功耗边缘设备上的 SNN 在线学习提供了一种高效、时间局部、性能接近 BPTT 的训练方法。
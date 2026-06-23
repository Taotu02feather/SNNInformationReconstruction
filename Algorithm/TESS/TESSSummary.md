# TESS：可扩展的时间与空间局部脉冲神经网络学习规则（中文完整整理）

## 1. 论文基本信息

- **论文标题**：TESS: A Scalable Temporally and Spatially Local Learning Rule for Spiking Neural Networks
- **中文标题**：TESS：一种可扩展的时间与空间局部脉冲神经网络学习规则
- **作者**：Marco P. E. Apolinario、Kaushik Roy、Charlotte Frenkel
- **单位**：Purdue University；Delft University of Technology
- **arXiv 版本**：arXiv:2502.01837v1，2025 年 2 月 3 日
- **关键词**：Spiking Neural Networks；Local Learning Rule；On-device Learning

---

## 2. 摘要核心内容

随着边缘设备上低功耗推理与训练需求增加，研究者需要同时具备可扩展性和能效的学习算法。SNN 具有事件驱动计算和处理复杂时空动态的能力，因此适合低功耗推理；但在资源受限设备上训练 SNN 仍然困难，主要原因是传统基于误差反向传播的方法具有很高的计算和内存需求。

论文提出 **TESS**，一种时间与空间都局部的 SNN 学习规则。TESS 受以下生物机制启发：

- eligibility traces；
- spike-timing-dependent plasticity（STDP）；
- neural activity synchronization。

TESS 的核心目标是同时解决：

1. **时间信用分配**；
2. **空间信用分配**。

与 BPTT 不同，TESS 只依赖每个神经元内部局部可得的信号，不需要跨时间反传，也不需要跨层误差反传。因此：

- 内存复杂度随神经元数量线性增长；
- 计算复杂度也随神经元数量线性增长；
- 两者都与时间步数无关。

实验表明，TESS 在 IBM DVS Gesture、CIFAR10-DVS、时间版本 CIFAR10、时间版本 CIFAR100 等边缘视觉任务上，能够达到接近 BPTT 的性能。其中在 CIFAR10-DVS 上相差约 1.4 个准确率点，而在其他数据集上能达到与 BPTT 相当的性能。

---

# 3. 引言

## 3.1 边缘设备上的学习需求

随着低功耗电子设备普及和深度神经网络发展，将智能部署到边缘设备成为重要趋势。

传统方式是：

1. 在云端离线训练 DNN；
2. 将训练好的模型部署到边缘设备。

但这种方式并不适合所有场景。例如：

- 数据隐私要求高；
- 设备需要实时适应环境；
- 无法频繁连接云端；
- 需要设备端持续学习。

这些场景需要 **on-device learning**，也就是在设备端训练或适配模型。

## 3.2 SNN 的优势与训练困难

SNN 因以下特性适合低功耗边缘推理：

- 二值脉冲激活；
- 事件驱动计算；
- 时空信息处理；
- 稀疏活动。

但是，要在边缘设备上训练 SNN，需要解决两个信用分配问题：

1. **Temporal credit assignment**：时间维度上，过去神经活动如何影响当前损失。
2. **Spatial credit assignment**：空间维度上，各层各神经元如何对误差负责。

BPTT 可以解决二者，但开销高。对于 $L$ 层、每层 $n$ 个神经元、时间步数为 $T$ 的 SNN，BPTT 的复杂度为：

$$
\begin{aligned}
\text{Time complexity:}&\quad O(T L n^2) \\
\text{Memory complexity:}&\quad O(T L n)
\end{aligned}
$$

由于复杂度依赖 $T$，BPTT 不适合低功耗设备上的 on-device learning。

---

## 3.3 现有局部学习方法

### 空间信用分配方法

已有方法包括：

- Feedback Alignment（FA）；
- Direct Feedback Alignment（DFA）；
- Direct Random Target Projection（DRTP）；
- Forward-forward 类方法；
- Local Learning Rule inspired by neural activity synchronization（LLS）。

这些方法尝试减少或替代误差反向传播，但有些存在：

- 收敛慢；
- 难以扩展到深层网络；
- 大数据集上性能下降明显。

### 时间信用分配方法

三因子学习规则使用 eligibility traces 保存神经活动历史，是处理时间信用分配的常见方法。

典型方法包括：

- e-prop；
- ETLP；
- OSTTP；
- S-TLLR；
- OTTT。

e-prop、ETLP、OSTTP 等方法虽然时间局部，但通常需要 $O(Ln^2)$ 内存，随突触数量增长，对深层网络昂贵。

S-TLLR 将内存降为 $O(Ln)$，且与时间步无关，但仍依赖跨层误差反向传播进行空间信用分配，因此只具备时间局部性，不具备完整空间局部性。

---

## 3.4 TESS 的定位

TESS 的目标是进一步克服 S-TLLR 的限制，实现：

- 时间局部；
- 空间局部；
- 低复杂度；
- 可扩展到深层 SNN；
- 适合边缘设备在线学习。

TESS 使用 eligibility traces 解决时间信用分配，并使用由固定 basis vectors 产生的局部学习信号解决空间信用分配。

它不需要全局误差跨层反向传播。

---

## 3.5 图 1 的含义

论文图 1 比较了三类学习规则：

### (a) 非局部学习方法，例如 BPTT

- 误差同时沿时间和空间反向传播；
- 解决时间与空间信用分配；
- 但不局部，开销高。

### (b) 时间局部方法

- 使用 eligibility traces 解决时间信用分配；
- 学习信号仍需要跨层传播；
- 时间局部，但空间不局部。

### (c) 完全局部方法，即 TESS

- 使用 eligibility traces 解决时间信用分配；
- 学习信号由每层局部生成；
- 时间和空间都局部。

---

# 4. 论文贡献

论文贡献包括：

1. **提出 TESS**：一种新的可扩展 SNN 学习规则，结合 eligibility traces、STDP、neural activity synchronization，只依赖局部信号，适合低功耗设备端学习。
2. **线性复杂度**：TESS 具有 $O(Ln)$ 内存复杂度和 $O(LCn)$ 计算复杂度，可训练 VGG-9 等较深架构。
3. **性能接近 BPTT**：在 IBM DVS Gesture、CIFAR10、CIFAR100 上达到与 BPTT 相当的准确率，在 CIFAR10-DVS 上仅下降约 1.4 个准确率点，同时显著降低资源开销。

---

# 5. 背景知识

## 5.1 LIF 模型

论文采用 Leaky Integrate-and-Fire（LIF）神经元模型：

$$
u_i^{(l)}[t] = \gamma \left( u_i^{(l)}[t-1] - v_{th} o_i^{(l)}[t-1] \right) + \sum_j W_{ij}^{(l)} o_j^{(l-1)}[t]
$$

$$
o_i^{(l)}[t] = \Theta \left( u_i^{(l)}[t] - v_{th} \right)
$$

其中：

- $u_i^{(l)}[t]$：第 $l$ 层第 $i$ 个神经元在时间 $t$ 的膜电位；
- $W_{ij}^{(l)}$：从第 $l-1$ 层第 $j$ 个神经元到第 $l$ 层第 $i$ 个神经元的权重；
- $\gamma$：泄露因子；
- $v_{th}$：阈值；
- $\Theta$：Heaviside 函数；
- $o_i^{(l)}[t]$：二值输出脉冲。

当膜电位达到阈值时，神经元发放脉冲，并通过 subtractive reset 减去 $v_{th}$。

---

## 5.2 SNN 的梯度优化

给定数据集：

$$
\mathcal{D} = \{(x, y^*)_i\}_{i=1}^N
$$

SNN 参数为：

$$
W = \{W^{(l)}\}_{l=1}^L
$$

目标是最小化损失：

$$
W := \arg\min_W \mathcal{L}(\mathcal{D}; W)
$$

常规梯度下降更新：

$$
W^{(l)} := W^{(l)} - \eta \frac{d\mathcal{L}}{dW^{(l)}}
$$

BPTT 计算梯度：

$$
\frac{d\mathcal{L}}{dW^{(l)}} = \sum_{t=1}^{T} \left[ \frac{\partial\mathcal{L}}{\partial u^{(l)}[t]} \right] \left[ \frac{\partial u^{(l)}[t]}{\partial W^{(l)}} \right]
$$

由于 SNN 具有递归状态，$\frac{\partial\mathcal{L}}{\partial u^{(l)}[t]}$ 依赖模型完整历史：

$$
\frac{\partial\mathcal{L}}{\partial u^{(l)}[t]} = \frac{\partial\mathcal{L}}{\partial o^{(l)}[t]} \cdot \frac{\partial o^{(l)}[t]}{\partial u^{(l)}[t]} + \frac{\partial\mathcal{L}}{\partial u^{(l)}[t+1]} \cdot \frac{\partial u^{(l)}[t+1]}{\partial u^{(l)}[t]}
$$

其中：

- $\frac{\partial\mathcal{L}}{\partial o^{(l)}[t]}$ 需要后续层信息；
- $\frac{\partial\mathcal{L}}{\partial u^{(l)}[t+1]} \cdot \frac{\partial u^{(l)}[t+1]}{\partial u^{(l)}[t]}$ 依赖时间历史。

因此 BPTT 既不是空间局部，也不是时间局部。

---

## 5.3 三因子学习规则

三因子学习规则认为，突触更新由三个因素共同决定：

1. 突触前活动；
2. 突触后活动；
3. 自上而下的学习信号。

资格迹定义为：

$$
e_{ij}^{(l)}[t] = \beta e_{ij}^{(l)}[t-1] + f(o_i^{(l)}[t]) \, g(o_j^{(l-1)}[t])
$$

权重更新为：

$$
\Delta W_{ij} = \sum_t m_i[t] \, e_{ij}^{(l)}[t]
$$

其中：

- $e_{ij}$：eligibility trace；
- $m_i[t]$：learning signal；
- $\beta$：衰减因子；
- $f$、$g$：突触后和突触前活动函数。

问题是：传统 eligibility trace 往往需要为每个突触保存状态，内存复杂度为 $O(Ln^2)$，难以扩展到深层网络。

---

# 6. TESS 方法

TESS 是一种三因子学习规则，专门解决 SNN 的时间与空间信用分配问题。它有两个核心组成部分：

1. 用 eligibility traces 实现时间信用分配；
2. 用 locally generated learning signals 实现空间信用分配。

---

## 6.1 时间信用分配：低内存 eligibility traces

传统 eligibility trace：

$$
e_{ij}[t] = \beta e_{ij}[t-1] + f(\text{post}) \, g(\text{pre})
$$

需要为每个突触保存 $e_{ij}$，内存为 $O(n^2)$。

TESS 采用与 S-TLLR 类似的思路，设置：

$$
\beta = 0
$$

即使用 instantaneous eligibility traces，从而只需独立追踪突触前和突触后活动 trace，把内存降为 $O(n)$。

TESS 使用两个资格迹：

- $e_{\text{pre}}$：基于 causal relation；
- $e_{\text{post}}$：基于 non-causal relation。

它们分别对应 STDP 中的因果项和非因果项。

---

## 6.2 causal eligibility trace

首先定义突触前活动的低通滤波 trace：

$$
q^{(l)}[t] = \lambda_{\text{pre}} q^{(l)}[t-1] + o^{(l-1)}[t]
$$

其中：

- $q^{(l)}[t]$：突触前活动 trace；
- $\lambda_{\text{pre}}$：指数衰减因子；
- $o^{(l-1)}[t]$：前一层输入脉冲。

然后 causal eligibility trace 为：

$$
e_{\text{pre}}^{(l)}[t] = \alpha_{\text{pre}} \Psi(u^{(l)}[t]) \otimes q^{(l)}[t]
$$

其中：

- $\Psi(u)$ 是 secondary activation function；
- $\otimes$ 表示外积；
- $\alpha_{\text{pre}}$ 控制资格迹幅度，实验中设为 1。

这里 $\Psi(u)$ 类似 surrogate gradient 的作用。

---

## 6.3 non-causal eligibility trace

TESS 还追踪膜电位激活函数的历史：

$$
h^{(l)}[t] = \lambda_{\text{post}} h^{(l)}[t-1] + \Psi(u^{(l)}[t-1])
$$

然后 non-causal eligibility trace 为：

$$
e_{\text{post}}^{(l)}[t] = \alpha_{\text{post}} h^{(l)}[t] \otimes o^{(l-1)}[t]
$$

其中：

- $h^{(l)}[t]$：post-synaptic activity 的历史 trace；
- $\lambda_{\text{post}}$：衰减因子；
- $\alpha_{\text{post}}$ 控制是否加入 non-causal term。

$\alpha_{\text{post}}$ 的含义：

- $\alpha_{\text{post}} = +1$：正向加入 non-causal term；
- $\alpha_{\text{post}} = -1$：负向加入 non-causal term；
- $\alpha_{\text{post}} = 0$：不使用 non-causal term。

---

## 6.4 空间信用分配：局部生成学习信号

传统三因子学习规则虽然可以用 eligibility trace 解决时间信用分配，但仍需要一个 top-down learning signal 来解决空间信用分配。

很多方法用 BP 或 DFA 生成这个信号，因此仍不是完全空间局部。

TESS 使用 **Learning Signal Generation（LSG）**，在每一层局部生成学习信号，不需要跨层反向传播。

过程如下：

1. 将每层输出脉冲 $o^{(l)}[t]$ 投影到 $C$ 维任务子空间：

$$
B^{(l)} o^{(l)}[t]
$$

2. 应用函数 $f(\cdot)$：

- 分类任务中，$f$ 是 softmax；
- 回归任务中，$f$ 是 identity。

3. 与目标 $y^*$ 比较得到局部误差信号：

$$
f(B^{(l)} o^{(l)}[t]) - y^*
$$

4. 再投影回当前层，得到调制学习信号：

$$
m^{(l)}[t] = {B^{(l)}}^T \left( f(B^{(l)} o^{(l)}[t]) - y^* \right)
$$

这个学习信号完全由当前层输出、固定投影矩阵和标签生成，因此是空间局部的。

---

## 6.5 固定二值矩阵 B 的设计

TESS 中 $B^{(l)}$ 是固定二值矩阵。每一列对应一个 square wave function。

这种设计的优点：

- 帮助同一层神经元活动同步；
- 为不同类别分配不同空间频率；
- 让任务相关信息分布到层内；
- 不同类别投影近似正交，减少干扰；
- square wave 简单，硬件实现高效。

该机制来自 neural activity synchronization 的启发。

---

## 6.6 权重更新

TESS 的 causal 权重更新：

$$
\Delta W_{\text{pre}}^{(l)}[t] = \left( m^{(l)}[t] \odot \alpha_{\text{pre}} \Psi(u^{(l)}[t]) \right) \otimes q^{(l)}[t]
$$

non-causal 权重更新：

$$
\Delta W_{\text{post}}^{(l)}[t] = \left( m^{(l)}[t] \odot \alpha_{\text{post}} h^{(l)}[t] \right) \otimes o^{(l-1)}[t]
$$

总权重更新为：

$$
\Delta W^{(l)}[t] = \Delta W_{\text{pre}}^{(l)}[t] + \Delta W_{\text{post}}^{(l)}[t]
$$

这正是三因子学习形式：

$$
\text{weight update} = \text{learning signal} \times \text{eligibility trace}
$$

其中学习信号 $m[t]$ 是局部生成的，而 eligibility trace 由局部 pre/post 神经活动计算。

---

## 6.7 TESS 算法流程

对每一层 $l$：

输入：

- $o^{(l-1)}$：该层输入；
- $B$：固定二值矩阵；
- $\beta$：阈值；
- $\eta$：学习率；
- $t_l$：开始生成学习信号的时间步。

初始化：

$$
u^{(l)}[0] = 0,\quad h^{(l)}[0] = 0,\quad q^{(l)}[0] = 0
$$

对每个时间步：

1. 根据公式 (7) 更新 $h^{(l)}[t]$；
2. 根据 LIF 公式更新膜电位与输出脉冲；
3. 根据公式 (5) 更新 $q^{(l)}[t]$；
4. 若 $t \geq t_l$：
   - 根据公式 (9) 计算局部学习信号 $m^{(l)}[t]$；
   - 根据公式 (10)、(11)、(12) 计算权重更新。
5. 最后累计所有时间步更新权重：

$$
W^{(l)} = W^{(l)} + \eta \sum_{t=t_l}^{T} \Delta W^{(l)}[t]
$$

---

# 7. 计算与内存复杂度

## 7.1 内存需求

BPTT 内存：

$$
\text{Mem}_{\text{BPTT}} = T \sum_l n^{(l)}
$$

S-TLLR 内存：

$$
\text{Mem}_{\text{S-TLLR}} = 2 \sum_l n^{(l)}
$$

TESS 内存：

$$
\text{Mem}_{\text{TESS}} = 2 \sum_l n^{(l)}
$$

TESS 与 S-TLLR 内存相当，但不需要 BPTT 那样保存所有时间步状态。

其中 factor 2 来自：

- causal trace $q$；
- non-causal trace $h$。

如果 $\alpha_{\text{post}} = 0$，可不保存 non-causal trace，内存还可进一步下降。

---

## 7.2 计算需求

论文重点比较生成学习信号的 MAC 操作。

BPTT：

$$
\text{MAC}_{\text{BPTT}} = T \sum_l n^{(l)} \times n^{(l-1)}
$$

S-TLLR：

$$
\text{MAC}_{\text{S-TLLR}} = (T - t_l) \sum_l n^{(l)} \times n^{(l-1)}
$$

TESS：

$$
\text{MAC}_{\text{TESS}} = (T - t_l) \sum_l 2 \times n^{(l)} \times C
$$

其中 $C$ 是类别数或回归变量维度。因为通常 $C \ll n$，TESS 的计算量显著低于 BPTT 与 S-TLLR。

大致降低因子约为：

$$
n / C
$$

---

## 7.3 与其他方法复杂度比较

| 方法 | 内存复杂度 | 时间复杂度 | 时间局部 | 空间局部 |
|---|---|---|---|---|
| BPTT | $O(TLn)$ | $O(TLn^2)$ | 否 | 否 |
| e-prop | $O(Ln^2)$ | $O(Ln^2)$ | 是 | 否 |
| OSTL | $O(Ln^2)$ | $O(Ln^2)$ | 是 | 否 |
| ETLP | $O(Ln^2)$ | $O(LCn)$ | 是 | 是 |
| OSTTP | $O(Ln^2)$ | $O(LCn)$ | 是 | 是 |
| OTTT | $O(Ln)$ | $O(Ln^2)$ | 是 | 否 |
| S-TLLR | $O(Ln)$ | $O(Ln^2)$ | 是 | 否 |
| TESS | $O(Ln)$ | $O(LCn)$ | 是 | 是 |

TESS 的优势是同时具备：

- 线性内存复杂度；
- 低时间复杂度；
- 时间局部；
- 空间局部。

---

# 8. 实验设置

## 8.1 数据集

论文使用四个数据集：

1. CIFAR10；
2. CIFAR100；
3. IBM DVS Gesture；
4. CIFAR10-DVS。

### CIFAR10 与 CIFAR100

- 图像输入被呈现给 SNN 6 个时间步；
- 数据增强包括：
  - zero-padding 到更大尺寸；
  - random crop 到 32×32；
  - cutout；
  - random horizontal flipping；
  - normalization。

### CIFAR10-DVS

- 事件累积为 10 个 event frames；
- resize 到 48×48；
- 数据增强包括 padding=4 的 random crop；
- 之后归一化。

### IBM DVS Gesture

- 序列长度不同；
- 切分为 1.5 秒样本；
- 累积为 20 个 event frames；
- 每个 frame 表示 75 ms；
- resize 到 32×32；
- padding=4 后 random crop。

---

## 8.2 训练细节

实验模型：VGG-9。

训练设置：

- Adam 优化器；
- 学习率 0.001；
- 训练 200 epochs；
- 若验证准确率连续 5 个 epoch 无提升，则学习率减半；
- $\lambda_{\text{pre}} = 0.5$；
- $\lambda_{\text{post}} = 0.2$；
- $\alpha_{\text{pre}} = 1$；
- LIF leak factor $\gamma = 0.5$；
- 阈值 $v_{th} = 0.6$；
- secondary activation function：

$$
\Psi(u) = 0.3 \cdot \max(1 - |u - v_{th}|, 0)
$$

权重在每个时间步更新：

$$
t_l = 0
$$

---

# 9. 实验结果

## 9.1 non-causal term 消融实验

论文测试 $\alpha_{\text{post}} \in \{-1, 0, +1\}$。

| 数据集 | T | $\alpha_{\text{post}}=-1$ | $\alpha_{\text{post}}=0$ | $\alpha_{\text{post}}=+1$ |
|---|---|---|---|---|
| CIFAR10-DVS | 10 | 75.00±0.69% | 75.00±0.65% | 74.36±0.87% |
| DVS Gesture | 20 | 98.56±0.41% | 98.33±0.57% | 98.56±0.31% |
| CIFAR10 | 6 | 89.93±0.31% | 91.99±0.19% | 92.55±0.16% |
| CIFAR100 | 6 | 62.49±1.05% | 68.19±0.55% | 70.00±0.34% |

结论：

- 除 CIFAR10-DVS 外，$\alpha_{\text{post}}=+1$ 在其他数据集上表现更好；
- 正向加入 non-causal term 可带来 0.23 到 1.81 个准确率点提升；
- $\alpha_{\text{post}}=-1$ 只在 IBM DVS Gesture 上带来提升；
- 加入 $\alpha_{\text{post}}$ 需要保存 $h[t]$，会增加内存；
- 因此 TESS 在性能和内存之间存在可调权衡。

---

## 9.2 与其他方法性能比较

### CIFAR10-DVS

| 方法 | 模型 | 局部学习 | T | Batch | 准确率 | MAC | Memory |
|---|---|---|---|---|---|---|---|
| BPTT baseline | VGG-9 | 否 | 10 | 64 | 76.40±0.66% | 13589.59×10⁶ | 25.50 MB |
| S-TLLR baseline | VGG-9 | 时间局部 | 10 | 64 | 75.14±1.37% | 13589.59×10⁶ | 5.10 MB |
| TESS | VGG-9 | 完全局部 | 10 | 64 | 75.00±0.65% | 22.15×10⁶ | 2.55 MB |

TESS 相比 BPTT 下降约 1.4 个准确率点，但显著降低 MAC 和内存。

### IBM DVS Gesture

| 方法 | 模型 | 局部学习 | T | Batch | 准确率 | MAC | Memory |
|---|---|---|---|---|---|---|---|
| BPTT baseline | VGG-9 | 否 | 20 | 16 | 97.95±0.68% | 12079.69×10⁶ | 22.69 MB |
| S-TLLR baseline | VGG-9 | 时间局部 | 20 | 16 | 98.48±0.37% | 12079.69×10⁶ | 2.26 MB |
| TESS | VGG-9 | 完全局部 | 20 | 16 | 98.56±0.31% | 22.65×10⁶ | 2.26 MB |

TESS 在 IBM DVS Gesture 上取得最佳结果，超过 BPTT 和 S-TLLR baseline。

### CIFAR10

| 方法 | 模型 | 局部学习 | T | Batch | 准确率 | MAC | Memory |
|---|---|---|---|---|---|---|---|
| BPTT baseline | VGG-9 | 否 | 6 | 128 | 92.55±0.06% | 3623.90×10⁶ | 6.83 MB |
| S-TLLR baseline | VGG-9 | 时间局部 | 6 | 128 | 91.88±0.28% | 3623.90×10⁶ | 2.27 MB |
| TESS | VGG-9 | 完全局部 | 6 | 128 | 92.55±0.16% | 5.48×10⁶ | 2.27 MB |

TESS 与 BPTT baseline 准确率相同，但计算量大幅下降。

### CIFAR100

| 方法 | 模型 | 局部学习 | T | Batch | 准确率 | MAC | Memory |
|---|---|---|---|---|---|---|---|
| BPTT baseline | VGG-9 | 否 | 6 | 128 | 69.28±0.37% | 3624.18×10⁶ | 6.83 MB |
| S-TLLR baseline | VGG-9 | 时间局部 | 6 | 128 | 68.00±0.71% | 3624.18×10⁶ | 2.27 MB |
| TESS | VGG-9 | 完全局部 | 6 | 128 | 70.00±0.34% | 17.64×10⁶ | 2.27 MB |

TESS 在 CIFAR100 上超过 BPTT baseline 和 S-TLLR baseline。

---

## 9.3 总体实验结论

论文指出：

- 在 CIFAR10-DVS 上，TESS 相比 BPTT baseline 只下降约 1.4 个准确率点；
- 在 IBM DVS Gesture、CIFAR10、CIFAR100 上，TESS 与 BPTT 相当，甚至更好；
- TESS 相比 BPTT 降低了约 205 到 661 倍 MAC；
- TESS 相比 BPTT 降低了约 3 到 10 倍内存；
- TESS 是完全局部学习方法，仍可达到接近非局部方法的性能。

---

# 10. 与 S-TLLR 的关系

TESS 可看作在 S-TLLR 基础上的进一步推进。

S-TLLR：

- 解决时间信用分配；
- 使用 STDP 风格 causal 和 non-causal traces；
- 显存 $O(Ln)$；
- 仍需要 BP 进行空间信用分配；
- 时间局部但空间不完全局部。

TESS：

- 保留 S-TLLR 风格的低内存 eligibility traces；
- 引入 LSG 局部学习信号；
- 不再需要跨层 BP；
- 同时具备时间局部和空间局部；
- 计算复杂度从 $O(Ln^2)$ 降为 $O(LCn)$。

---

# 11. 结论与展望

论文提出 TESS，一种时间与空间都局部的 SNN 学习规则，旨在满足边缘设备上低功耗、可扩展训练的需求。

TESS 的主要价值在于：

1. 将 BPTT 的内存复杂度从 $O(TLn)$ 降到 $O(Ln)$；
2. 将时间复杂度从 $O(TLn^2)$ 降到 $O(LCn)$；
3. 用 eligibility traces 进行时间信用分配；
4. 用局部生成学习信号进行空间信用分配；
5. 避免全局信息流和跨层误差反传；
6. 在多个视觉任务上达到与 BPTT 接近的性能。

实验结果说明，TESS 是一种适合低功耗设备端学习的可扩展方案，尤其适用于需要实时适应、资源受限、强调时空局部性的 SNN 硬件系统。

---

# 12. 最核心总结

TESS 的核心思想可以浓缩为：

> 用 eligibility traces 解决时间信用分配，
> 用固定二值投影矩阵局部生成学习信号解决空间信用分配，
> 从而让 SNN 训练同时实现时间局部和空间局部。

与 BPTT、S-TLLR 的区别：

> - BPTT：时间不局部，空间不局部，性能好但开销高。
> - S-TLLR：时间局部，但空间仍依赖误差反传。
> - TESS：时间局部，空间局部，性能接近 BPTT，资源开销极低。

因此，TESS 的意义不只是降低内存，而是把 SNN 训练推进到更适合片上学习和边缘设备在线学习的完全局部学习范式。
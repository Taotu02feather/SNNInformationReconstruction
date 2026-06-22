以下整理依据论文摘要、方法、实验表、附录与影响声明。    

# NDOT：基于神经元动力学的脉冲神经网络在线训练方法

## 1. 论文基本信息

**标题**：NDOT: Neuronal Dynamics-based Online Training for Spiking Neural Networks
**中文标题**：NDOT：基于神经元动力学的脉冲神经网络在线训练
**会议**：ICML 2024
**作者**：Haiyan Jiang、Giulia De Masi、Huan Xiong、Bin Gu
**代码地址**：[https://github.com/HaiyanJiang/SNN-NDOT](https://github.com/HaiyanJiang/SNN-NDOT)

---

## 2. 摘要核心内容

脉冲神经网络（SNN）因其在神经形态计算中的低能耗和快速推理特性而受到关注。然而，深层 SNN 的高效训练仍然困难，主要原因是二值脉冲生成激活函数不可微。

现有常用方法是 surrogate gradient（SG）与 backpropagation through time（BPTT）结合。BPTT 需要沿时间展开计算图，并在所有时间步存储中间信息，因此显存消耗巨大，也不满足在线训练要求。

论文提出 **Neuronal Dynamics-based Online Training（NDOT）**。NDOT 在梯度计算中利用基于神经元动力学的连续时间依赖关系，将完整梯度分解为：

* temporal gradient：时间梯度；
* spatial gradient：空间梯度。

这样可以实现 forward-in-time learning，即按时间向前进行在线学习，而不需要沿时间反向传播。

论文还借助 **Follow-the-Regularized-Leader（FTRL）** 算法解释 NDOT 的直觉：FTRL 显式利用历史信息，而 NDOT 通过神经元动力学隐式捕获历史时间依赖。

实验表明，NDOT 在 CIFAR-10、CIFAR-100 和 CIFAR10-DVS 上，在较少时间步内取得了优越性能。

---

## 3. 引言

SNN 被称为第三代神经网络。与传统 ANN 相比，SNN 更接近生物神经系统，通过脉冲序列在层间传递信息。

SNN 的优势来自脉冲的二值、事件驱动特性：

* 推理时可避免大量乘法运算；
* 在 TrueNorth、Loihi、Tianjic 等神经形态硬件上具有低能耗潜力；
* 适合快速推理和边缘计算。

但 SNN 从零开始监督训练非常困难，因为脉冲生成过程是离散且不可微的。

---

## 4. SNN 训练方法背景

### 4.1 ANN-to-SNN 转换

ANN-to-SNN 转换方法先训练 ANN，再将权重映射到 SNN。

优点：

* ANN 更容易训练；
* 可利用成熟 ANN 训练技术。

缺点：

* 通常需要较长仿真时间步才能达到 ANN 性能；
* 延迟和能耗增加；
* 对神经形态数据不一定合适。

### 4.2 基于梯度的直接训练

直接训练通常使用 surrogate gradient 解决脉冲不可微问题。

前向传播时使用 Heaviside 函数产生脉冲；反向传播时，用可微 surrogate function 替代 Heaviside 函数。

常见做法是把 SNN 看作二值 RNN，然后使用 BPTT。

问题：

* 必须保存所有时间步的中间状态；
* 显存消耗随时间步线性增长；
* 对 ImageNet 等大规模任务训练成本很高；
* 不支持真正的 forward-in-time 在线学习。

### 4.3 基于脉冲表示的方法

另一类方法使用 spike representation，例如 weighted firing rate，将时间维度的信息合并。

这些方法把 SNN 的脉冲表示与类似 ANN 的映射联系起来，从而像训练 ANN 一样训练 SNN。

优点：

* 不必显式沿时间反向传播；
* 梯度计算更接近 ANN 形式。

缺点：

* 通常需要更多时间步；
* rate-based 表示会增加延迟和能耗。

### 4.4 在线训练方法

已有在线训练方法试图降低训练显存，并保持生物合理的在线学习特性。

代表方法包括：

* RTRL；
* UORO；
* KF-RTRL；
* SnAp；
* FPTT；
* OTTT；
* SLTT。

但这些方法往往没有精确捕获时间维度梯度，导致相对于 BPTT 可能存在性能劣势。

NDOT 的目标就是在保持在线训练的同时，更准确地捕获时间依赖。

---

## 5. 论文贡献

论文主要贡献如下：

1. **提出 NDOT**

   * 直接训练 SNN；
   * 支持 forward-in-time learning；
   * 不需要大量训练显存；
   * 训练显存与时间步数无关，保持常数级。

2. **准确捕获时间梯度**

   * NDOT 使用神经元动力学捕获时间维度的 temporal gradients；
   * 再结合跨层的 spatial gradients；
   * 由此得到完整梯度。

3. **用 FTRL 提供直觉解释**

   * FTRL 显式利用历史信息；
   * NDOT 通过神经元动力学隐式捕获历史时间依赖；
   * 实验中 FTRL-OTTT 的提升支持了这一解释。

4. **实验性能优越**

   * 在 CIFAR-10、CIFAR-100、CIFAR10-DVS 上表现优于多种 SOTA 方法；
   * 在小时间步下仍取得高准确率。

---

## 6. 预备知识

## 6.1 脉冲神经网络

SNN 中的脉冲神经元受生物神经元启发。神经元将输入脉冲序列积分到膜电位 `u(t)` 中，当膜电位超过阈值 `Vth` 时发放脉冲，并在发放后重置膜电位。

脉冲用 0 和 1 表示：

* 0：静息；
* 1：激活。

这种二值脉冲序列在层间传递信息，使 SNN 能在神经形态芯片上进行事件驱动计算。

---

## 6.2 LIF 神经元模型

论文使用 Leaky Integrate-and-Fire（LIF）模型描述膜电位动力学：

```text
τ du(t)/dt = -(u(t) - urest) + I(t),  u(t) < Vth
```

其中：

* `τ`：时间常数；
* `I(t)`：输入电流；
* `Vth`：发放阈值；
* `urest`：静息电位，通常为 0。

当 `u(t)` 达到阈值 `Vth` 时，神经元在时间 `tf` 发放脉冲，并将膜电位重置为静息电位。

脉冲序列可写为：

```text
s(t) = Σ_tf δ(t - tf)
```

---

## 6.3 离散 LIF 模型

考虑输入电流：

```text
I_i[t] = Σ_j W_ij s_j[t]
```

离散形式为：

```text
u_i[t+1] = λ(u_i[t] - Vth s_i[t]) + Σ_j W_ij s_j[t+1]

s_i[t] = H(u_i[t] - Vth)
```

其中：

* `H(·)` 是 Heaviside 阶跃函数；
* `s_i[t]` 是第 i 个神经元在时间步 t 的脉冲；
* `λ` 是泄露常数；
* 重置操作通过减去阈值 `Vth` 实现。

---

## 7. 先前 SNN 训练方法

## 7.1 Spike Representation

论文关注 weighted firing rate。

定义 weighted firing rate：

```text
a[t] = (Σ_{τ=1}^t λ^{t-τ} s[τ]) / (Σ_{τ=1}^t λ^{t-τ})
```

定义 weighted average input：

```text
x̄[t] = (Σ_{τ=1}^t λ^{t-τ} x[τ]) / (Σ_{τ=1}^t λ^{t-τ})
```

对于多层前馈 SNN，相邻层之间可建立基于 weighted firing rate 的闭式映射：

```text
a^l[T] ≈ σ((1/Vth) W^l a^{l-1}[T])
```

其中：

```text
σ(x) = min(max(0, x), 1)
```

该函数是离散情况下的 clamp function。

在这种视角下，weighted firing rate 类似 ANN 中的激活值，因此可基于层间 spike representation 映射计算梯度。

---

## 7.2 BPTT with SG

对多层前馈 SNN，LIF 递推为：

```text
u^l[t+1] = λ(u^l[t] - Vth s^l[t]) + W^l s^{l-1}[t+1]
```

BPTT 将该递推沿时间展开，并通过时间反向传播。

梯度包含两部分：

1. 当前时间步对权重的直接影响；
2. 历史时间步通过膜电位动态传递到当前时间步的影响。

其中不可微项：

```text
∂s^l[i] / ∂u^l[i]
```

通常被 surrogate gradient 替代，例如矩形函数或 sigmoid 函数的导数。

BPTT 的问题是：

* 需要存储整个时间展开计算图；
* 显存开销随时间步增加；
* 不满足在线训练需求。

---

## 8. FTRL：Follow-the-Regularized-Leader

论文引入 FTRL 是为了帮助解释 NDOT 为什么有效。

在在线优化中，每一轮 t 得到一个样本 `z_t`，根据当前权重 `w_t` 计算即时损失 `ℓ_t(w_t; z_t)` 和梯度 `g_t`，然后更新到 `w_{t+1}`。

正则化随机学习的目标函数为：

```text
f_t(w) = ℓ_t(w) + Ψ(w)
```

其中：

* `ℓ_t(w)` 是任务损失；
* `Ψ(w)` 是正则化项。

FTRL-Proximal 更新规则为：

```text
w_{t+1} = argmin_w (g_{1:t} w + tΨ(w) + 1/2 Σ_{s=1}^t ||Q_s^{1/2}(w - w_s)||_2^2)
```

FTRL 的关键思想是显式使用历史信息。论文用这个思想解释 NDOT：NDOT 虽然没有显式添加历史正则项，但通过神经元动力学隐式捕获了历史时间依赖。

---

# 9. NDOT 方法

## 9.1 从神经元动力学观察时间依赖

LIF 神经元包含三个基本过程：

1. charging：来自突触前输入 `s(t)` 的充电；
2. leakage：膜电位从 `u(t)` 到 `u(t+1)` 的泄露或衰减；
3. firing：膜电位通过脉冲生成过程 `u(t) -> s(t)`。

从 `u(t)` 到 `u(t+1)` 的时间信息流包含这三类成分。

BPTT 中的离散时间依赖可写为：

```text
ε^l[t] =
∂u^l[t+1]/∂u^l[t]
+
∂u^l[t+1]/∂s^l[t] · ∂s^l[t]/∂u^l[t]
```

这表示 `u^l[t+1]` 对 `u^l[t]` 的离散时间敏感性。

如果对 `∂s/∂u` 使用 surrogate derivative，就得到常见的 SG-BPTT。

如果不使用 surrogate derivative，并近似认为脉冲路径的时间依赖为 0，就得到 OTTT 的退化形式。

---

## 9.2 连续时间依赖表示

NDOT 的关键是用神经元动力学得到连续时间依赖表示。

论文将从 `u(t)` 到 `u(t+1)` 的完整时间依赖记为：

```text
u(t) ⇝ u(t+1)
```

并用隐式函数表示：

```text
u(t+1) = Im(u(t))
```

通过链式法则：

```text
du(t+1)/dt = ∂Im/∂u(t) · ∂u(t)/∂t
```

定义连续时间依赖：

```text
e(t) = ∂u(t+1)/∂u(t)
     = u'(t+1) ⊘ u'(t)
```

其中 `⊘` 表示逐元素除法。

结合 LIF 动力学，可得到：

```text
e(t) = (u(t+1) - I(t+1)) / (u(t) - I(t))
```

在离散时间步和不同层上：

```text
e^l[t] = (u^l[t+1] - I^l[t+1]) / (u^l[t] - I^l[t])
```

进一步结合 SNN 递推式，得到：

```text
e^l[t] = (u^l[t] - Vth s^l[t]) / (u^l[t-1] - Vth s^l[t-1])
```

这就是 NDOT 用来捕获时间依赖的核心量。

---

## 9.3 NDOT 的核心推导

NDOT 用 `e[t]` 替代 BPTT 中的离散时间依赖 `ε[t]`。

为了实现 forward-in-time 在线学习，论文将完整梯度分解为：

* 时间部分；
* 空间部分。

定义 temporal component gradient：

```text
â^{l-1}[t]
=
∂u^l[t]/∂W^l
+
Σ_{k<t} Π_{i=k}^{t-1} e^{l-1}[i] ⊙ ∂u^l[k]/∂W^l
```

进一步定义：

```text
P^l_{k,t} = Π_{i=k}^{t-1} e^l[i]
```

则权重梯度可写为：

```text
∇_{W^l} L
=
Σ_{t=1}^T g_u^l[t] (
s^{l-1}[t] + Σ_{k<t} P^{l-1}_{k,t} ⊙ s^{l-1}[k]
)^T
```

其中：

```text
g_u^l[t] = (∂L/∂s^l[t] · ∂s^l[t]/∂u^l[t])^T
```

是空间梯度。

定义 tracked temporal gradient，也称 presynaptic activities：

```text
â^{l-1}[t]
=
s^{l-1}[t] + Σ_{k<t} P^{l-1}_{k,t} ⊙ s^{l-1}[k]
```

递推形式为：

```text
â^{l-1}[t]
=
e^{l-1}[t-1] ⊙ â^{l-1}[t-1] + s^{l-1}[t]
```

因此，在每个时间步，只要得到空间梯度 `g_u^l[t]`，就能直接计算完整梯度：

```text
∇_{W^l} L = g_u^l[t] · â^{l-1}[t]^T
```

这避免了沿时间反向传播。

---

## 9.4 NDOT 算法流程

一次 NDOT 训练迭代如下：

1. 输入训练样本 `(x, y)` 和时间步 `T`。
2. 对每个时间步 `t = 1, 2, ..., T`：

   * 对每一层执行前向传播；
   * 更新膜电位 `u^l[t]`；
   * 生成脉冲 `s^l[t]`；
   * 计算神经元动力学时间依赖 `e^l[t]`；
   * 更新 temporal gradient `â^l[t]`。
3. 从输出层向输入层计算空间梯度 `g_u^l[t]`。
4. 计算即时梯度：

```text
∇_{W^l} L = g_u^l[t] · â^{l-1}[t]^T
```

5. 使用梯度优化器更新参数。

---

## 9.5 空间梯度与时间梯度

在完整梯度：

```text
∇_{W^l} L = g_u^l[t] · â^{l-1}[t]^T
```

中：

* `g_u^l[t]` 是 spatial gradient，表示层与层之间的空间维度梯度；
* `â^{l-1}[t]` 是 temporal gradient，表示时间维度梯度。

这种解耦使 NDOT 能够在时间上向前计算梯度。

---

## 9.6 瞬时损失

传统 SNN 分类通常基于平均发放率：

```text
(1/T) Σ_{t=1}^T s^N[t]
```

对应离线损失为：

```text
L_off = ℓ((1/T)Σ_{t=1}^T s^N[t], y)
```

该损失依赖所有时间步，无法支持在线学习。

NDOT 使用瞬时损失：

```text
L[t] = (1/T) ℓ(s^N[t], y)
```

总损失为：

```text
L = Σ_{t=1}^T L[t]
```

当 `ℓ` 是凸函数时，总损失是 `L_off` 的上界。

论文进一步结合交叉熵损失和均方误差损失：

```text
L[t] = (1 - α) · ℓ_CE(s^N[t], y) + α · ℓ_MSE(s^N[t], y)
```

该混合损失来自先前 OTTT 和 TET 的观察：加入 MSE loss 可以提高 SNN 训练准确率。

---

## 10. NDOT 背后的直觉

## 10.1 FTRL-OTTT

论文提出 FTRL-OTTT 作为对比和解释工具。

FTRL-OTTT 分两阶段：

1. 第一阶段：

   * 使用较小时间步，例如 `T = 2`；
   * 训练更长 epoch；
   * 得到较优权重 `W_hat`。

2. 第二阶段：

   * 使用较大目标时间步，例如 `T = 4`；
   * 优化即时损失；
   * 同时让权重接近第一阶段得到的 `W_hat`。

FTRL-OTTT 损失为：

```text
L_hat[t] = L[t] + ρ ||W - W_hat||_2^2
```

或：

```text
L_hat[t] = L[t] + ρ ||W - W_hat||_1
```

这里 `W_hat` 总结了历史信息，并作为 FTRL 正则项显式加入。

---

## 10.2 NDOT 的隐式历史信息利用

NDOT 没有显式添加 FTRL 正则项，但它通过神经元动力学中的 `e[t]` 递推捕获历史时间依赖。

因此：

* FTRL-OTTT 显式利用历史信息；
* NDOT 通过神经元动力学隐式利用历史信息。

实验表明，FTRL-OTTT 相比 OTTT 有明显提升，而 NDOT 与 FTRL-OTTT 有相似甚至更好的表现。这说明精确捕获时间依赖确实有助于提高在线 SNN 训练性能。

---

# 11. 实验设置

## 11.1 数据集

论文实验使用：

* CIFAR-10；
* CIFAR-100；
* CIFAR10-DVS。

### CIFAR-10

包含：

* 10 类彩色图像；
* 50,000 个训练样本；
* 10,000 个测试样本；
* 每张图像大小为 `32 × 32 × 3`。

预处理：

* 全局均值和标准差归一化；
* 随机裁剪；
* 水平翻转；
* cutout 数据增强。

SNN 第一层输入直接使用像素值，可看作实值输入电流。

### CIFAR-100

与 CIFAR-10 类似，但有 100 类。

包含：

* 50,000 个训练样本；
* 10,000 个测试样本。

预处理与 CIFAR-10 相同。

### CIFAR10-DVS

CIFAR10-DVS 是 CIFAR-10 的神经形态版本，由 Dynamic Vision Sensor 转换得到。

特点：

* 10,000 个样本；
* 是原 CIFAR-10 的六分之一；
* spike trains 有两个通道：ON-event 和 OFF-event；
* 原始像素维度扩展到 `128 × 128`；
* 划分为 9000 个训练样本和 1000 个测试样本。

预处理：

* 将事件聚合为 2 个时间步；
* 空间分辨率插值到 `48 × 48`；
* 使用随机裁剪；
* 用所有时间步的全局均值和标准差归一化。

---

## 11.2 网络结构和训练设置

实验使用 VGG 网络结构：

```text
64C3-128C3-AP2-256C3-256C3-AP2-512C3-512C3-AP2-512C3-512C3-GAP-FC
```

优化器：

* SGD；
* 无 weight decay；
* 初始学习率 0.1；
* 使用 cosine decay 衰减到 0。

LIF 参数：

```text
Vth = 1
λ = 0.5
```

---

## 11.3 NDOT 的两种更新策略

### NDOT_O

在线实时更新：

* 每个时间步计算梯度后立即更新参数；
* 然后进入下一个时间步。

### NDOT_A

累计时间更新：

* 在 T 个时间步内累计梯度；
* 最后统一更新参数。

---

## 11.4 不使用 Batch Normalization

许多 BPTT+SG 方法使用沿时间维度的 Batch Normalization 来获得低延迟高性能。

但时间维 BN 需要在前向过程中统计所有时间步的均值和方差，因此：

* 不兼容在线梯度；
* 增加显存消耗；
* 破坏 forward-in-time 训练。

NDOT 使用 scaled Weight Standardization（WS）替代 BN：

```text
W_hat_ij = γ · (W_ij - μ_Wi) / (σ_Wi √N)
```

其中 `γ` 是缩放参数。

---

# 12. 实验结果

## 12.1 CIFAR-10

| 方法         | 类型              | 网络         |  时间步 |   准确率 |
| ---------- | --------------- | ---------- | ---: | ----: |
| SPIKE-NORM | ANN-to-SNN      | VGG-16     | 2500 | 91.55 |
| ReLuTS     | ANN-to-SNN      | VGG-16     |   16 | 92.29 |
| QCFS       | ANN-to-SNN      | VGG-16     |    4 | 93.96 |
| SlipReLu   | ANN-to-SNN      | ResNet-18  |    1 | 93.11 |
| BNTT       | BPTT            | VGG-9      |   20 | 90.30 |
| TEBN       | BPTT            | VGG-9      |    4 | 92.81 |
| tdBN       | BPTT            | ResNet-19  |    4 | 92.92 |
| SLTT       | BPTT            | ResNet-18  |    6 | 94.59 |
| LTL        | Tandem Learning | VGG-11     |   16 | 93.20 |
| OTTT       | Forward-in-time | VGG-11（WS） |    6 | 93.73 |
| NDOT_O     | Forward-in-time | VGG-11（WS） |    6 | 94.89 |
| NDOT_O     | Forward-in-time | VGG-11（WS） |    4 | 94.79 |
| NDOT_O     | Forward-in-time | VGG-11（WS） |    2 | 94.44 |
| NDOT_O     | Forward-in-time | VGG-11（WS） |    1 | 94.28 |
| NDOT_A     | Forward-in-time | VGG-11（WS） |    6 | 94.90 |
| NDOT_A     | Forward-in-time | VGG-11（WS） |    4 | 94.86 |
| NDOT_A     | Forward-in-time | VGG-11（WS） |    2 | 94.41 |

结论：

* NDOT 在 CIFAR-10 上超过所有已有方法；
* 即使 `T = 1`，NDOT_O 也达到 94.28%；
* `T = 1` 的 NDOT 仍比 `T = 16` 的 LTL 高 1.08%。

---

## 12.2 CIFAR-100

| 方法         | 类型              | 网络         |  时间步 |   准确率 |
| ---------- | --------------- | ---------- | ---: | ----: |
| SPIKE-NORM | ANN-to-SNN      | VGG-16     | 2500 | 70.90 |
| SlipReLu   | ANN-to-SNN      | ResNet-18  |    4 | 74.89 |
| Hybrid     | Hybrid          | VGG-11     |  125 | 67.87 |
| BNTT       | BPTT            | VGG-11     |   50 | 66.60 |
| TEBN       | BPTT            | VGG-11     |    4 | 74.37 |
| TET        | BPTT            | ResNet-19  |    4 | 74.62 |
| LTL        | Tandem Learning | VGG-11     |   16 | 72.63 |
| OTTT       | Forward-in-time | VGG-11（WS） |    6 | 71.11 |
| FTRL-OTTT  | Forward-in-time | VGG-11（WS） |    6 | 75.89 |
| FTRL-OTTT  | Forward-in-time | VGG-11（WS） |    4 | 74.95 |
| FTRL-OTTT  | Forward-in-time | VGG-11（WS） |    2 | 74.55 |
| NDOT_O     | Forward-in-time | VGG-11（WS） |    6 | 76.61 |
| NDOT_O     | Forward-in-time | VGG-11（WS） |    4 | 76.18 |
| NDOT_O     | Forward-in-time | VGG-11（WS） |    2 | 75.27 |
| NDOT_O     | Forward-in-time | VGG-11（WS） |    1 | 73.24 |
| NDOT_A     | Forward-in-time | VGG-11（WS） |    6 | 76.47 |
| NDOT_A     | Forward-in-time | VGG-11（WS） |    4 | 76.12 |
| NDOT_A     | Forward-in-time | VGG-11（WS） |    2 | 75.01 |

结论：

* CIFAR-100 上 NDOT 优势更明显；
* 与 OTTT 在 `T = 6` 的 71.11% 相比，NDOT_O 达到 76.61%；
* 提升超过 5%；
* 说明 NDOT 对复杂数据集尤其有效；
* FTRL-OTTT 也显著优于 OTTT，支持“历史信息/时间依赖”很重要这一观点。

---

## 12.3 CIFAR10-DVS

| 方法      | 类型              | 网络          | 时间步 |   准确率 |
| ------- | --------------- | ----------- | --: | ----: |
| NeuNorm | BPTT            | 7-layer CNN |  40 | 60.50 |
| BNTT    | BPTT            | 7-layer CNN |  20 | 63.20 |
| tdBN    | BPTT            | ResNet-19   |  10 | 67.80 |
| TEBN    | BPTT            | 7-layer CNN |  10 | 75.10 |
| PLIF    | BPTT            | 7-layer CNN |  20 | 74.80 |
| SLTT    | BPTT            | VGG-11      |  10 | 77.30 |
| OTTT    | Forward-in-time | VGG-11（WS）  |  10 | 77.10 |
| NDOT_O  | Forward-in-time | VGG-11（WS）  |  10 | 77.50 |
| NDOT_A  | Forward-in-time | VGG-11（WS）  |  10 | 77.40 |

结论：

* NDOT 在神经形态数据集 CIFAR10-DVS 上也优于 OTTT；
* NDOT_O 达到 77.50%；
* NDOT_A 达到 77.40%。

---

## 12.4 训练显存开销

NDOT 的重要优势是无需沿时间维度反向传播。

因此：

* 对固定 batch size，训练显存不随时间步增加；
* 与 BPTT 相比，避免了保存所有时间步中间状态的开销；
* 可以在相同计算资源下使用更大 batch size 加速训练。

论文在 CIFAR-100 上使用 VGG，batch size 从 32 到 256，时间步从 2 到 30 变化。实验显示：在同一 batch size 下，NDOT 的 GPU 显存基本保持常数，不随时间步增长。

---

## 12.5 损失中超参数 α 的消融实验

论文在 DVS-CIFAR10 上固定 `T = 6`，使用 NDOT_O，测试不同 α。

|        α |  0.0 | 0.05 |  0.1 |  0.2 |  0.3 |  0.4 |
| -------: | ---: | ---: | ---: | ---: | ---: | ---: |
| Accuracy | 74.2 | 75.1 | 75.6 | 75.1 | 74.8 | 74.7 |

|        α |  0.5 |  0.6 |  0.7 |  0.8 |  0.9 |  1.0 |
| -------: | ---: | ---: | ---: | ---: | ---: | ---: |
| Accuracy | 73.8 | 74.1 | 73.1 | 73.7 | 72.6 | 71.0 |

结论：

* α = 0.1 时准确率最高，为 75.6%；
* α 太小或太大都会降低性能；
* 交叉熵和 MSE 的平衡组合最有效。

---

# 13. 附录内容

## 13.1 NDOT 详细推导

附录详细推导了 NDOT 从 BPTT 梯度到 Eq. (10) 的过程。

BPTT 梯度中的时间部分包含：

```text
Π ε^l[i]
```

NDOT 用神经元动力学中的连续时间依赖：

```text
e^l[i]
```

替代离散时间依赖：

```text
ε^l[i]
```

并通过逐元素乘法保证维度匹配。

最终得到：

```text
∇_{W^l} L
=
Σ_t g_u^l[t] (â^{l-1}[t])^T
```

其中：

```text
â^{l-1}[t]
=
e^{l-1}[t-1] ⊙ â^{l-1}[t-1] + s^{l-1}[t]
```

---

## 13.2 数值稳定策略

计算：

```text
e^{l-1}[t]
=
(u^{l-1}[t] - Vth s^{l-1}[t])
/
(u^{l-1}[t-1] - Vth s^{l-1}[t-1])
```

时，分母可能为 0。

论文实现中采用数值稳定策略：

* 当分母为 0 时，先判断 `u^{l-1}[t-1]` 的符号；
* 然后使用 clamp 函数将值限制在 `[-λ, λ]` 范围内；
* 这样可以缓解数值不稳定，提高鲁棒性。

---

## 13.3 实现细节

NDOT 每个时间步计算即时梯度：

```text
∂L[t]/∂W^l = g_u^l[t] â^{l-1}[t]
```

有两种实现：

* NDOT_O：每个时间步立即更新参数；
* NDOT_A：累计 T 个时间步梯度后再更新参数。

论文不使用时间维度 BN，而使用 scaled Weight Standardization 替代。

---

## 13.4 更多实验结果

### CIFAR-100

| 方法     | 时间步 |   准确率 |
| ------ | --: | ----: |
| NDOT_A |   6 | 76.47 |
| NDOT_A |   4 | 76.12 |
| NDOT_A |   2 | 75.01 |
| NDOT_A |   1 | 73.24 |
| OTTT_A |   6 | 71.11 |
| OTTT_O |   6 | 71.11 |
| NDOT_O |   6 | 76.61 |
| NDOT_O |   4 | 76.18 |
| NDOT_O |   2 | 75.27 |
| NDOT_O |   1 | 73.24 |

### DVS-CIFAR10

| 方法     | 时间步 |   准确率 |
| ------ | --: | ----: |
| NDOT_A |  10 |  77.4 |
| NDOT_A |   8 |  77.3 |
| NDOT_A |   6 |  76.0 |
| NDOT_A |   4 |  74.9 |
| NDOT_A |   2 |  71.1 |
| OTTT_A |  10 | 76.30 |
| OTTT_O |  10 | 77.10 |
| NDOT_O |  10 |  77.5 |
| NDOT_O |   8 |  77.3 |
| NDOT_O |   6 |  75.6 |
| NDOT_O |   4 |  74.1 |
| NDOT_O |   2 |  71.1 |

结论：

* NDOT_A 在 CIFAR-100 上即使 `T = 1` 也达到 73.24%，超过 OTTT_A 在 `T = 6` 的 71.11%；
* 在 `T = 6` 时，NDOT_A 达到 76.47%，比 OTTT_A 提升 5.35%；
* 在 DVS-CIFAR10 上，NDOT_A 在 `T = 8` 时达到 77.3%，超过 OTTT_A 在 `T = 10` 的 76.30%；
* NDOT 在不同数据集和不同时间步下都表现出鲁棒性和通用性。

---

# 14. 影响声明

论文目标是推进机器学习领域，特别是低能耗 AI 算法。

作者认为该研究主要关注高性能、低延迟 SNN 的直接训练，没有明显负面影响。

由于 SNN 具有节能特性，随着应用增加，它们可能在边缘计算中变得重要。

---

# 15. 结论

论文解决的是 SNN 中 forward-in-time 训练问题，目标是避免沿时间维度展开计算图，从而降低显存开销。

NDOT 的核心思想是：

1. 从神经元动力学出发构造连续时间依赖 `e[t]`；
2. 用 `e[t]` 捕获时间维度梯度；
3. 将完整梯度分解为空间梯度和时间梯度；
4. 在每个时间步独立计算完整梯度；
5. 实现无需时间反向传播的在线训练。

与 OTTT 相比，NDOT 不只是近似忽略部分时间依赖，而是通过神经元动力学更准确地捕获时间依赖。

FTRL 的分析说明，历史信息对在线训练非常重要。FTRL-OTTT 显式利用历史权重，而 NDOT 通过神经元动力学隐式捕获历史时间信息。

实验表明，NDOT 在 CIFAR-10、CIFAR-100 和 CIFAR10-DVS 上都表现优越，尤其在 CIFAR-100 等复杂数据集上相比 OTTT 有显著提升，同时保持训练显存对时间步数的常数级开销。

因此，NDOT 是一种兼顾低显存、在线训练、低延迟和高性能的 SNN 训练方法。

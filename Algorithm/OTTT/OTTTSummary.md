
# Online Training Through Time for Spiking Neural Networks 中文整理

## 1. 论文基本信息

论文题目：Online Training Through Time for Spiking Neural Networks  
中文：面向脉冲神经网络的在线时间训练方法  
会议：NeurIPS 2022  
作者：Mingqing Xiao、Qingyan Meng、Zongpeng Zhang、Di He、Zhouchen Lin  
代码：https://github.com/pkuxmq/OTTT-SNN

## 2. 核心问题

脉冲神经网络（SNN）具有类脑、事件驱动和低能耗优势，适合神经形态硬件。但监督训练困难，主要因为脉冲生成函数不可微。

现有主流方法有两类：

1. BPTT + surrogate gradients  
   - 优点：能在很少时间步内取得高性能，例如 4–6 个时间步。
   - 缺点：训练时需要保存沿时间展开的计算图，显存消耗随时间步线性增长；代理梯度的优化方向缺少清晰理论保证；不符合生物学习和神经形态芯片上的在线学习。

2. 基于脉冲表示的方法  
   - 通过 firing rate、weighted firing rate 或 spike time 建立类似 ANN 的闭式映射。
   - 优点：优化方向更清晰。
   - 缺点：通常需要较多时间步，延迟和能耗较高；也不是在线训练。

## 3. 论文提出的方法：OTTT

论文提出 Online Training Through Time（OTTT），即“在线时间训练”。

OTTT 的目标是同时实现：

- 低延迟；
- 高性能；
- 训练显存不随时间步增长；
- 前向时间在线学习；
- 与脉冲表示方法建立理论联系；
- 具有三因子 Hebbian 学习形式，具备潜在生物合理性和片上学习可能。

## 4. 主要贡献

1. 提出 OTTT，用于 SNN 在线训练。它通过追踪突触前活动解耦 BPTT 的时间依赖，使训练显存对时间步保持常数级。
2. 从理论上连接 OTTT 梯度与基于脉冲表示的梯度，并证明在前馈网络和循环网络条件下，OTTT 能提供类似的下降方向。
3. 证明 OTTT 的更新规则可写成三因子 Hebbian 学习形式，连接了 BPTT+SG、脉冲表示训练和生物学习规则。
4. 在 CIFAR-10、CIFAR-100、ImageNet、CIFAR10-DVS、DVS128-Gesture 等数据集上验证了方法有效性。

## 5. 预备知识

### 5.1 LIF 脉冲神经元

论文采用常用的 Leaky Integrate-and-Fire（LIF）模型。神经元维护膜电位 u，输入电流使膜电位积分，当膜电位超过阈值 Vth 时发放脉冲，并将膜电位重置。

连续形式：

τm du/dt = -(u - urest) + R · I(t)

离散形式：

$$ u_i[t+1] = \lambda(u_i[t] - V_{th} s_i[t]) + \sum_j w_{ij} s_j[t] + b_i $$
$$ s_i[t+1] = H(u_i[t+1] - V_{th}) $$

其中：

- H 是 Heaviside 阶跃函数；
- s_i[t] 是第 i 个神经元在时间 t 的脉冲；
- λ < 1 是泄露项；
- Vth 是阈值。

### 5.2 基于脉冲表示的训练

论文重点关注 weighted firing rate。给定收敛的加权平均输入，weighted firing rate 可收敛到由 clamp 或 ReLU 类函数决定的映射。

前馈网络中，相邻层可近似写成：

$$ a^{l+1}[T] \approx \sigma\left(\frac{1}{V_{th}}(W^l a^l[T] + b^{l+1})\right) $$

循环网络中，weighted firing rate 可收敛到隐式平衡点：

$$ a^* = \sigma\left(\frac{1}{V_{th}}(W a^* + F x^* + b)\right) $$

梯度可通过闭式映射或隐式微分计算。

### 5.3 BPTT + SG

BPTT 将 SNN 沿时间展开并反向传播。由于 Heaviside 函数不可微，训练时通常用代理梯度近似脉冲函数导数。

问题在于：

- 显存随时间步增加；
- surrogate gradient 为什么有效缺少理论解释；
- 不是在线学习。

## 6. OTTT 的推导

### 6.1 解耦时间依赖

BPTT 的时间依赖来自脉冲神经元膜电位在不同时间步之间的递推。

OTTT 的关键思想是：

- 不在时间依赖路径中使用 surrogate derivative；
- Heaviside 函数导数几乎处处为 0，因此重置路径中的时间反传项可近似忽略；
- 剩下的时间依赖主要由膜电位泄露项 λ 决定；
- 因此可以通过追踪突触前活动来避免时间反传。

定义追踪的突触前活动：

$$ \hat{a}^l[t] = \sum_{\tau \leq t} \lambda^{t-\tau} s^l[\tau] $$

递推形式：

$$ \hat{a}^l[t+1] = \lambda \hat{a}^l[t] + s^l[t+1] $$

于是每个时间步的权重梯度可写为：

$$ \nabla_{W^l} L[t] = g_u^{l+1}[t] \cdot \hat{a}^l[t]^T $$

其中 g_u^{l+1}[t] 是当前时间步对膜电位的梯度。

这样，OTTT 不需要沿时间反向传播，也不需要保存完整时间展开计算图。

### 6.2 瞬时损失

传统 SNN 常使用 firing rate 损失：

$$ L_{\text{fr}} = L\left(\frac{1}{T}\sum_t s^N[t], y\right) $$

该损失依赖所有时间步，不能支持在线梯度。

OTTT 使用瞬时损失：

$$ L[t] = \frac{1}{T} L(s^N[t], y) $$

总损失：

$$ L = \sum_t L[t] $$

当 L 是凸函数，例如交叉熵时，这个总损失是 firing-rate 损失的上界。

### 6.3 前馈与循环网络

对于前馈网络，OTTT 梯度可通过每层当前时间步的误差信号和追踪的突触前活动计算。

对于带反馈连接的循环网络，OTTT 也适用。任意从层 li 到层 lj 的连接权重梯度可写成：

$$ \nabla_{W^{li \to lj}} L[t] = g_u^{lj}[t] \cdot \hat{a}^{li}[t]^T $$

## 7. OTTT 与脉冲表示方法的理论联系

论文证明 OTTT 梯度与基于 spike representation 的梯度形式相似。

### 7.1 关键观察

OTTT 中追踪的突触前活动 â[t] 与 weighted firing rate a[t] 密切相关。

基于脉冲表示的方法使用最终时间 T 的 weighted firing rate，而 OTTT 使用每个时间步的即时 â[t]。

### 7.2 surrogate derivative 的重新解释

论文指出，OTTT 中使用的 surrogate derivative 并不是像 BPTT+SG 那样作为 Heaviside 函数的伪导数，而是用于近似 spike representation 映射函数 σ 的导数。

这给 surrogate derivative 提供了更清晰的理论解释。

### 7.3 前馈网络定理

在 Assumption 1、Vth = 1，并且 weighted firing rate 的时间误差足够小的条件下，OTTT 梯度与 spike representation 梯度内积大于 0。

含义：

- OTTT 的负梯度方向可作为 spike representation 优化问题的下降方向；
- 随机误差可以看作随机优化中的噪声。

### 7.4 循环网络定理

对于带反馈连接的网络，论文借鉴隐式模型和固定点优化的理论。

若平衡映射的 Jacobian 满足一定收缩条件，且输入与神经元 firing rate 的时间误差足够小，则 OTTT 梯度与基于隐式微分的 spike representation 梯度也具有正内积。

含义：

- OTTT 在循环 SNN 中也能提供合理下降方向；
- 理论可推广到多层网络和任意反馈结构。

## 8. 与三因子 Hebbian 学习规则的关系

OTTT 的单个连接更新可写为：

$$ \nabla_{W_{ij}} L[t] = \hat{a}_i[t] \cdot f(u_j[t]) \cdot \delta_j[t] $$

其中：

- $\hat{a}_i[t]$：突触前活动；
- $f(u_j[t])$：突触后活动变化率，即 surrogate derivative；
- $\delta_j[t]$：全局误差信号或调制信号。

这正是三因子 Hebbian 学习形式：

1. 突触前因子；
2. 突触后因子；
3. 全局调制因子。

因此 OTTT 具有潜在生物合理性，并可能用于神经形态芯片上的在线学习。

## 9. 实现细节

OTTT 有两种更新方式：

### 9.1 OTTT_O

每个时间步计算瞬时梯度后立即更新参数。

### 9.2 OTTT_A

在 T 个时间步内累计梯度，然后统一更新参数。

### 9.3 不使用 Batch Normalization

现有 BPTT+SG 方法常使用沿时间维度的 Batch Normalization 来提升低时间步性能，但这需要跨时间统计，破坏在线性并增加显存。

OTTT 不使用 BN，而采用 scaled Weight Standardization（sWS）替代。

sWS 对权重标准化：

$$ W_{\text{hat}}_{ij} = \gamma \cdot \frac{W_{ij} - \mu_{W_{i,.}}}{\sigma_{W_{i,.}} \sqrt{N}} $$

论文针对 SNN 的 Heaviside 激活推导 γ，取 γ ≈ 2.74。

## 10. 实验设置

数据集：

- CIFAR-10
- CIFAR-100
- ImageNet
- CIFAR10-DVS
- DVS128-Gesture
- Fashion-MNIST（附录循环结构实验）

网络：

- CIFAR-10、CIFAR-100、CIFAR10-DVS、DVS128-Gesture：VGG 结构
- ImageNet：NF-ResNet-34
- Fashion-MNIST：400 个循环脉冲神经元

默认参数：

- Vth = 1
- λ = 0.5

训练细节：

- CIFAR-10、CIFAR-100、DVS-CIFAR10：SGD，momentum 0.9，300 epochs，batch size 128，初始学习率 0.1，cosine annealing。
- ImageNet：SGD，momentum 0.9，100 epochs，batch size 256，初始学习率 0.1，每 30 epoch 衰减 0.1。
- DVS128-Gesture：Adam，300 epochs，batch size 16，初始学习率 0.001。
- 实验基于 PyTorch，在一张 NVIDIA RTX 3090 GPU 上完成。

## 11. 实验结果

### 11.1 显存消耗

OTTT 的显存消耗与时间步无关，保持常数级。

BPTT 的显存随时间步线性增长。

在 CIFAR-10、VGG、batch size 128 的实验中，即使只有 6 个时间步，OTTT 也可比 BPTT 减少约 2–3 倍显存。

### 11.2 CIFAR-10

| 方法 | 网络 | 参数量 | 时间步 | 结果 |
|---|---:|---:|---:|---:|
| BPTT | VGG-sWS | 9.2M | 6 | 92.78±0.34%，最好 93.23% |
| OTTT_A | VGG-sWS | 9.2M | 6 | 93.52±0.06%，最好 93.58% |
| OTTT_O | VGG-sWS | 9.2M | 6 | 93.49±0.17%，最好 93.73% |
| ANN | VGG-sWS | 9.2M | N/A | 94.43% |

OTTT 在相同训练设置下优于 BPTT，且与 ANN 的差距约 0.7%。

### 11.3 CIFAR-100

| 方法 | 网络 | 参数量 | 时间步 | 结果 |
|---|---:|---:|---:|---:|
| BPTT | VGG-sWS | 9.3M | 6 | 69.06±0.07%，最好 69.15% |
| OTTT_A | VGG-sWS | 9.3M | 6 | 71.05±0.04%，最好 71.11% |
| OTTT_O | VGG-sWS | 9.3M | 6 | 71.05±0.06%，最好 71.11% |
| ANN | VGG-sWS | 9.3M | N/A | 73.19% |

OTTT 与 ANN 差距约 2.08%。

### 11.4 ImageNet

| 方法 | 网络 | 参数量 | 时间步 | 结果 |
|---|---:|---:|---:|---:|
| BPTT | ResNet-34-tdBN | 22M | 6 | 63.72% |
| OTTT_A | NF-ResNet-34 | 22M | 6 | 65.15% |
| OTTT_O | NF-ResNet-34 | 22M | 6 | 64.16% |

OTTT_A 在 ImageNet 上优于 BPTT。

### 11.5 CIFAR10-DVS

| 方法 | 网络 | 参数量 | 时间步 | 结果 |
|---|---:|---:|---:|---:|
| BPTT | VGG-sWS | 9.2M | 10 | 72.60±1.26%，最好 73.90% |
| OTTT_A | VGG-sWS | 9.2M | 10 | 76.27±0.05%，最好 76.30% |
| OTTT_O | VGG-sWS | 9.2M | 10 | 76.63±0.34%，最好 77.10% |

OTTT 在神经形态数据集 CIFAR10-DVS 上显著优于同设置 BPTT。

### 11.6 DVS128-Gesture

| 方法 | 网络 | 时间步 | 准确率 |
|---|---:|---:|---:|
| BPTT | VGG-sWS | 20 | 96.88% |
| OTTT_A | VGG-sWS | 20 | 96.88% |

虽然理论主要针对收敛输入，OTTT 在时间变化更强的 DVS128-Gesture 上也能达到与 BPTT 相同的高性能。

### 11.7 反馈连接实验

在 CIFAR-100 上，加入从最后特征层到第一特征层的反馈连接形成 VGG-F。

| 网络 | 参数量 | 结果 |
|---|---:|---:|
| VGG | 9.3M | 71.05±0.06%，最好 71.11% |
| VGG-F | 9.6M | 72.63±0.23%，最好 72.94% |

反馈连接提升了 OTTT 性能，并且附录显示 OTTT 从反馈连接获得的提升比 BPTT 更明显。

### 11.8 Batch size = 1 的在线训练

CIFAR-10、VGG、训练 20 epochs：

| 方法 | Batch size | 准确率 |
|---|---:|---:|
| OTTT_A / OTTT_O | 128 | 88.20% / 88.62% |
| OTTT_A / OTTT_O | 1 | 88.07% / 88.50% |

说明 OTTT 具备进一步实现“每次一个样本”的完全在线训练潜力。

### 11.9 推理时间步影响

ImageNet 上，用 6 个时间步训练的模型，在推理时增加时间步可继续提升性能：

- 1 step：58.92%
- 2 steps：62.63%
- 4 steps：64.52%
- 6 steps：65.15%
- 8 steps：65.23%
- 10 steps：65.35%
- 12 steps：65.40%

### 11.10 发放率与能耗

CIFAR-10 上，OTTT 模型前层 firing rate 更高、后层 firing rate 更低。整体 firing rate 约 0.19，6 个时间步中每个神经元平均约产生 1.1 个脉冲，说明能耗较低。

ImageNet 附录中，OTTT_A 模型整体 firing rate 约 0.24，6 个时间步平均每个神经元产生 1.46 个脉冲。若使用 2 个时间步，平均仅 0.48 个脉冲，但准确率下降约 2.5%。

### 11.11 完全循环结构实验

Fashion-MNIST 上使用 400 个循环脉冲神经元：

| 方法 | 网络 | 时间步 | 准确率 |
|---|---:|---:|---:|
| BPTT | R400 | 5 | 90.58% |
| OTTT_A | R400 | 5 | 90.36% |
| OTTT_O | R400 | 5 | 90.40% |

在这个较简单的循环模型上，OTTT 与 BPTT 结果接近，BPTT 略好。

## 12. 附录中的重要内容

### 12.1 Eq. 6 推导

附录 A 推导了 spike representation 梯度形式，说明其与 OTTT 梯度具有相似结构。核心差别是：

- spike representation 梯度使用最终时间 T 的活动；
- OTTT 使用每个时间步的瞬时追踪活动。

### 12.2 多层网络时间记号说明

多层网络中，严格来说跨层传播存在突触延迟。论文为简化符号，统一使用时间步 t 表示每层对应的离散时间，但实际物理时间应考虑跨层延迟。

### 12.3 定理证明

附录 A 给出 Theorem 1 和 Theorem 2 的完整证明。

证明思路：

- 将 OTTT 梯度与 spike representation 梯度写成可比较形式；
- 使用 weighted firing rate 收敛误差较小的假设；
- 证明两种梯度内积为正；
- 从而说明 OTTT 梯度可作为对应优化问题的下降方向。

### 12.4 OTTT 伪代码

每次迭代：

1. 对每个时间步 t：
   - 前向更新膜电位；
   - 生成脉冲；
   - 更新追踪的突触前活动。
2. 从输出层向输入层计算当前时间步误差。
3. 计算瞬时梯度。
4. 若为 OTTT_O，则立即更新参数。
5. 若为 OTTT_A，则累计梯度，最后统一更新。

### 12.5 数据集与预处理

- CIFAR-10 / CIFAR-100：32×32 彩色图像，使用标准归一化、随机裁剪、水平翻转、cutout。
- ImageNet：随机 resize-crop 到 224×224，水平翻转，测试时 resize 到 256×256 后中心裁剪。
- CIFAR10-DVS：事件数据积累为 10 个时间步，空间分辨率插值到 48×48。
- DVS128-Gesture：事件数据整合为 20 帧。
- Fashion-MNIST：输入展平为 784 维，连接到 400 个循环脉冲神经元。

## 13. 局限性

OTTT 为了保持在线训练性质，限制了一些结构技术的使用，例如沿时间维度的 Batch Normalization。

论文使用 scaled Weight Standardization 替代 BN，但这可能需要额外正则化才能完全追上 BN 在 ANN 中的最佳性能。

未来需要探索更适合 SNN、同时兼容在线性质的新技术。

## 14. 社会影响

论文认为该工作主要关注 SNN 训练方法，没有直接负面社会影响。

潜在正面影响包括：

- 推动低能耗 SNN 模型发展；
- 减少 ANN 带来的巨大能耗；
- 帮助理解生物可行的神经网络训练；
- 缩小生物神经元与深度学习模型之间的差距。

## 15. 总结

OTTT 是一种面向 SNN 的在线训练方法。它从 BPTT+SG 出发，通过追踪突触前活动解耦时间依赖，使训练显存不随时间步增长。论文进一步将 OTTT 梯度与基于 spike representation 的梯度联系起来，并证明其在前馈和循环网络中可提供下降方向。

OTTT 同时具有三因子 Hebbian 学习形式，连接了三类原本相对分离的方法：

1. BPTT + surrogate gradient；
2. 基于 spike representation 的训练；
3. 生物合理的三因子 Hebbian 学习。

实验表明，OTTT 在少量时间步下即可在静态图像和神经形态数据集上取得优于或接近 BPTT 的性能，并显著降低训练显存。

# SNN 在线/本地学习：细节问题清单

> 本文档记录读论文、推导公式、设计实验过程中出现的细节问题。
> 分为「已解决」和「待解决」两类，用于后续跟踪。

---

## 一、符号与约定

### 1.1 LIF 复位形式

三种常见写法：

- **A**：$u_{t+1} = \lambda(u_t - V_{th}s_t) + I$
- **B**：$u_{t+1} = \lambda u_t - V_{th}s_t + I$
- **C**：$u_{t+1} = \lambda u_t(1-s_t) + I$

对应的时间依赖矩阵 $\epsilon^l[i]$：

| 形式 | $\epsilon^l[i]$ |
|---|---|
| A | $\lambda I - \lambda V_{th} D_{\Psi,i}$ |
| B | $\lambda I - V_{th} D_{\Psi,i}$ |
| C | $\lambda(1-s)I - \lambda u D_{\Psi,i}$ |

**为什么会影响推导**：三种形式的区别在复位项的位置。形式 A 的 reset 被 $\lambda$ 一起衰减，形式 B 的 reset 不衰减，形式 C 是乘性复位。这直接改变 $\epsilon^l[i]$ 里 reset 项的系数。

**本文约定**：统一用形式 A。理由是与 OTTT / NDOT / e-prop 的符号约定最接近，跨算法比较时不用换符号。

**状态**：已冻结。

---

### 1.2 全导 vs 偏导

链式法则中：

$$\frac{dL}{dW_{ji}^l} = \sum_t \frac{\partial L}{\partial s_j^l[t]} \cdot \frac{\partial s_j^l[t]}{\partial u_j^l[t]} \cdot \frac{du_j^l[t]}{dW_{ji}^l}$$

**为什么最后一项必须是全导**：$W_{ji}^l$ 通过两条路径影响 $u_j^l[t]$：

- 直接路径：$t$ 时刻的输入 $W_{ji}^l s_i^{l-1}[t]$ 进入 $u_j^l[t]$；
- 间接路径：$W_{ji}^l$ 影响 $u_j^l[t-1]$，再经过状态转移到 $u_j^l[t]$。

如果写偏导 $\partial u_j^l[t]/\partial W_{ji}^l$，右边直接等于 $s_i^{l-1}[t]$，**丢掉所有历史路径**，这就不是 BPTT。

**全导展开**：$du_j^l[t]/dW_{ji}^l = \sum_{\tau \le t} (\partial u_j^l[t]/\partial u_j^l[\tau]) \cdot (\partial u_j^l[\tau]/\partial W_{ji}^l)$。

注意展开后的 $\partial u_j^l[\tau]/\partial W_{ji}^l$ 是偏导，因为 $u_j^l[\tau]$ 对权重的直接依赖只通过 $\tau$ 时刻那一项。

**状态**：已冻结。

---

### 1.3 时间索引

$A_t \equiv \partial u^l[t]/\partial u^l[t-1]$，因此 $\epsilon^l[i] = \partial u^l[i+1]/\partial u^l[i]$ 内部所有 spike Jacobian 项用 $D_{\Psi,i}$，不用 $D_{\Psi,t}$。

**为什么**：$\epsilon^l[i]$ 的下标是 $i$，表示从时间步 $i$ 到 $i+1$ 的转移，所以 $\epsilon^l[i]$ 里出现的 $s^l[i]$ 对 $u^l[i]$ 求导，得到 $D_{\Psi,i}$。

**状态**：已冻结。

---

### 1.4 层索引映射

不同论文对 $W^l$ 定义不同：

- OTTT 原文：$W^l$ 表示 $l \to l+1$；
- NDOT 原文：$W^l$ 表示 $l-1 \to l$。

本文统一用 $W^{l \leftarrow l-1}$ 表示从 $l-1$ 到 $l$ 的权重。

**状态**：已冻结。

---

## 二、资格迹

### 2.1 资格迹的维护粒度

| 算法 | 粒度 | 维度 |
|---|---|---|
| e-prop | 每突触 | $\xi_{ji}^t \in \mathbb{R}$ |
| OTTT | 每层 | $\hat{a}^{l-1}[t] \in \mathbb{R}^{n^{(l-1)}}$ |
| NDOT | 每层 | $\hat{a}^{l-1}[t] \in \mathbb{R}^{n^{(l-1)}}$ |
| S-TLLR | 分解成两个神经元级 trace | $\text{tr}(x_j), \text{tr}(\Psi(u_i)) \in \mathbb{R}$ |
| TESS | 每层两个向量 | $q^{(l)} \in \mathbb{R}^{n^{(l-1)}}, h^{(l)} \in \mathbb{R}^{n^{(l)}}$ |

**为什么不同**：

- **e-prop 不能共享**：递推里用的是突触特有的 pre 脉冲 $z_i^t$，不同 $(i,j)$ 的历史不同；
- **OTTT/NDOT 可以共享**：递推里用的是整层脉冲向量 $s^l[t]$，同层内所有以该层为输入的突触共享同一个 trace；
- **S-TLLR/TESS 用外积即时构造**：不显式存储突触级 eligibility，只存两个神经元级 trace，更新权重时用外积生成。

**内存代价**：e-prop $O(Ln^2)$，OTTT/NDOT/S-TLLR/TESS $O(Ln)$。

**状态**：已冻结。

---

### 2.2 资格迹如何适配不同层的维度

外积自动适配：

$$\nabla_{W^{l \leftarrow l-1}} L[t] = g_{u^l}[t] \otimes \hat{a}^{l-1}[t]^\top$$

维度：

- $g_{u^l}[t] \in \mathbb{R}^{n^{(l)}}$（突触后维度）
- $\hat{a}^{l-1}[t] \in \mathbb{R}^{n^{(l-1)}}$（突触前维度）
- 外积：$\mathbb{R}^{n^{(l)} \times n^{(l-1)}}$，正好等于 $W^{l \leftarrow l-1}$ 的形状。

**不同层用自己的资格迹**：层 0 → 层 1 用 $\hat{a}^0$，层 1 → 层 2 用 $\hat{a}^1$。**不跨层共享**。

**状态**：已冻结。

---

### 2.3 $D_{\Psi,t}$ 是矩阵还是向量

$D_{\Psi,t} = \operatorname{diag}[\Psi'(u_j^l[t] - V_{th})]_j \in \mathbb{R}^{n^{(l)} \times n^{(l)}}$

**是矩阵，不是向量**。

**为什么**：它要参与矩阵运算 $\epsilon^l[i] = \lambda I - \lambda V_{th} D_{\Psi,i}$。$\lambda I$ 是 $n \times n$ 矩阵，$D_{\Psi,i}$ 也必须是 $n \times n$ 矩阵才能相减。

**物理含义**：第 $j$ 个对角元 $\Psi'(u_j^l[t] - V_{th})$ 衡量神经元 $j$ 对膜电位的敏感度。膜电位接近阈值时最大，远离阈值时快速衰减。

**状态**：已冻结。

---

### 2.4 TESS 中 $\odot$ 和 $\otimes$ 的区别

- $\odot$：逐元素乘法（Hadamard product），**同维度**输入，输出维度不变；
- $\otimes$：外积，**不同维度**输入，输出是两个维度的乘积。

TESS 的权重更新：

$$\Delta W^{(l)}[t] = \big(m^{(l)}[t] \odot \alpha_{\text{pre}} \Psi(u^{(l)}[t])\big) \otimes q^{(l)}[t] + \big(m^{(l)}[t] \odot \alpha_{\text{post}} h^{(l)}[t]\big) \otimes o^{(l-1)}[t]$$

**流程**：

1. 先用 $\odot$ 让每个神经元的 learning signal 调制它自己的 activation（仍然是向量，维度 $n^{(l)}$）；
2. 再用 $\otimes$ 把"突触后调制后的 activation"与"突触前 trace"组合，生成突触级权重更新（矩阵，维度 $n^{(l)} \times n^{(l-1)}$）。

**为什么要两个符号**：$\odot$ 用于"调制"（维度不变），$\otimes$ 用于"组合"（生成矩阵）。维度关系不同，不能混用。

**状态**：已冻结。

---

### 2.5 恒等式 $(m \odot a) \otimes b = m \odot (a \otimes b)$

这是 TESS 内存从 $O(n^2)$ 降到 $O(n)$ 的数学基础。

**证明**：

- 左边第 $(i,j)$ 元素：$(m_i \cdot a_i) \cdot b_j$
- 右边第 $(i,j)$ 元素：$m_i \cdot (a_i \cdot b_j)$
- 两者相等。

**意义**：TESS 不需要显式存储 $n \times n$ 的突触级资格迹 $e$，只需要存两个神经元级向量 $q, h$，更新权重时用外积即时构造 $e$。

**状态**：已冻结。

---

## 三、更新时机

### 3.1 两种更新模式

**Accumulated 模式**：

$$W \leftarrow W - \eta \sum_{t=1}^{T} \nabla_W L[t]$$

序列内 $W$ 不变，末尾更新一次。

**Online 模式**：

$$W \leftarrow W - \eta \nabla_W L[t]$$

每个时间步覆盖式更新。

**状态**：已冻结。

---

### 3.2 各算法的更新模式

| 算法 | 更新模式 | 说明 |
|---|---|---|
| BPTT | 天然 Accumulated | 序列末尾更新一次 |
| e-prop | Ideal 累积 / Online 每步 | 取决于 learning signal 用 ideal 还是近似 |
| OTTT | O 模式每步 / A 模式累积 | 两种都支持 |
| NDOT | O 模式每步 / A 模式累积 | 两种都支持 |
| S-TLLR | 每步更新 | 但只有最后几步有学习信号 |
| TESS | 累积末尾更新 | Algorithm 1 明确 |

**梯度保真度比较必须用 Accumulated 模式。**

**状态**：已冻结。

---

### 3.3 权重 $W$ 的存储

**两种模式都只存一份 $W$。**

- Accumulated：$W$ 序列内不变，末尾更新一次；
- Online：$W$ 每步覆盖，历史版本不保留。

内存里始终只有一份 $W$ 和（如果需要）一份梯度累积缓冲区。

**为什么不需要保存历史 $W$**：前向只依赖当前 $W$，梯度只在当前 $W$ 上算，更新是覆盖式的。

**状态**：已冻结。

---

### 3.4 为什么 Online 模式不能直接和 BPTT 比较梯度

Online 模式下：

- $t=1$ 的梯度在 $W_0$ 上算；
- $t=2$ 的梯度在 $W_1 = W_0 - \eta g_1$ 上算；
- $t=3$ 的梯度在 $W_2$ 上算；
- ...

$\sum_t g_t(W_{t-1})$ 不是同一个 $W$ 下的梯度总和，和 BPTT 的 $g$ 不在同一个评估点。

**后果**：如果强行比较，混淆了两部分——算法近似和评估点漂移。**无法归因**。

**Accumulated 模式消除混淆**：整段序列用同一个 $W_0$，$\sum_t g_t(W_0)$ 和 BPTT 的 $g$ 可以直接比较。

**状态**：已冻结。

---

## 四、算法理解

### 4.1 e-prop、OTTT、NDOT 的关系

| 算法 | 角色 | 关键操作 |
|---|---|---|
| e-prop | 分解框架 | 梯度 = 学习信号 × 资格迹 |
| OTTT | 时间近似 | 丢复位项，用固定 $\lambda$ |
| NDOT | 时间近似 | 用动态 $e^l[t]$ 替代 $\lambda$ |

三者共享统一形式 $\nabla_W E = \sum_t L_t \cdot e_t$，区别在资格迹的定义和粒度。

**核心公式**：

$$\text{BPTT}: \epsilon^l[i] = \lambda I - \lambda V_{th} D_{\Psi,i}$$
$$\text{OTTT}: \epsilon^l[i] \approx \lambda I$$
$$\text{NDOT}: \epsilon^l[i] \approx \operatorname{diag}(e^l[t])$$

**状态**：已冻结。

---

### 4.2 e-prop 的 eligibility 是否精确

**eligibility 是精确的**。它是 chain-rule rearrangement 后的局部项，可以由突触和 postsynaptic neuron 的局部 forward variables 计算。

**近似发生在 learning signal**：$L_j^t = dE/dz_j^t$ 需要未来信息，在线时用 $\hat{L}_j^t$ 近似。

**状态**：已冻结。

---

### 4.3 S-TLLR 的 non-causal 项

"non-causal" 是 STDP timing 意义上的 non-causal——当前 pre spike 与过去 post activity 的相关性。

**不访问未来**，仍是 online。

**公式**：

$$e_{ij}[t] = \alpha_{\text{pre}} \Psi(u_i[t]) \sum_{t'=0}^{t} \lambda_{\text{pre}}^{t-t'} x_j[t'] + \alpha_{\text{post}} x_j[t] \sum_{t'=0}^{t-1} \lambda_{\text{post}}^{t-t'} \Psi(u_i[t'])$$

第一项（因果）：当前 post × 过去 pre 迹。
第二项（非因果）：当前 pre × 过去 post 迹（减去当前）。

**状态**：已冻结。

---

### 4.4 S-TLLR 的 $\Psi$ 和 BPTT 的 $\Psi'_{\text{SG}}$

两者是不同对象：

- $\Psi$：secondary activation，作用于膜电位，生成连续突触后活动。用于资格迹的平滑；
- $\Psi'_{\text{SG}}$：spike surrogate derivative，替代不可导的 $\partial s/\partial u$。用于反向传播。

$$\Psi(u) \neq \Psi'_{\text{SG}}(u)$$

**状态**：已冻结。

---

### 4.5 TESS 和 S-TLLR 的核心区别

- S-TLLR：时间局部（STDP 迹），但空间仍需要 BP/DFA；
- TESS：时间局部（STDP 迹）+ 空间局部（本地 LSG）。

TESS 是 S-TLLR 的"空间也局部化"版本。

**关键**：S-TLLR 的 $\delta_i[t]$ 由跨层 BP/DFA 得到，TESS 的 $m^{(l)}[t]$ 完全在层内生成。

**状态**：已冻结。

---

### 4.6 TESS 的 B 矩阵为什么能替代跨层反传

来自 DFA 的思想：

- 用固定随机矩阵 $B^{(l)}$ 替代真实反向权重 $W^{(l+1)\top}$；
- 误差不需要精确反向，只需要方向正相关；
- 训练过程中前向权重会自适应去配合随机反馈（feedback alignment）。

**核心直觉**：不是"随机矩阵恰好对"，而是"网络会学会配合随机矩阵"。

**状态**：已冻结（详细证明需精读 DFA 原始论文）。

---

### 4.7 TESS 的 B 矩阵是固定的还是训练的

**固定的**。

好处：

- 硬件友好（不需要存储梯度）；
- 实验表明效果足够；
- 训练 $B$ 会引入额外的梯度路径，复杂度增加。

**状态**：已冻结。

---

## 五、时空解耦

### 5.1 解耦的本质

$$\nabla_W L = \underbrace{(\text{空间因子})}_{\text{独立算}} \times \underbrace{(\text{时间因子})}_{\text{独立算}}$$

两个因子独立计算，最后乘积组合。

**不是"互相不影响"**，而是"分别计算，最后组合"。

**状态**：已冻结。

---

### 5.2 解耦需要什么条件

**数学上的精确解耦无条件**——它是链式法则重排的恒等式。

**在线解耦需要四个条件**：

1. **隐藏状态局部**：$h_j^t$ 只依赖 $h_j^{t-1}$ 和外部输入；
2. **权重影响局部**：$W_{ji}$ 只直接影响 $h_j$；
3. **马尔可夫性**：$h_j^t$ 只依赖 $h_j^{t-1}$；
4. **代理梯度可用**：用光滑函数替代 $\partial s/\partial u$。

LIF / ALIF 满足前三条。

**如果违反**：比如跨神经元耦合动力学（$h_j^t$ 依赖 $h_k^{t-1}$），资格迹维度会从 $O(n)$ 回到 $O(n^2)$。

**状态**：已冻结。

---

### 5.3 解耦的代价

- 资格迹 $e_{ji}^t$：**精确保留**（可以前向递推）；
- 学习信号 $L_j^t$：**必须近似**（需要未来信息）。

$$\boxed{\text{解耦 = 时间因子精确保留，空间因子必须近似}}$$

**这就是为什么不同算法的差异集中在"怎么近似 $L_j^t$"**：OTTT/NDOT 用瞬时信号，e-prop 用反馈权重，TESS 用本地 LSG。

**状态**：已冻结。

---

## 六、空间局部化

### 6.1 Spatial BP 是什么

沿"层"方向反向传播误差，与 Temporal BP（沿时间）相对。

| 算法 | Temporal BP | Spatial BP |
|---|---|---|
| BPTT | 有 | 有 |
| OTTT / NDOT | 无 | 有 |
| e-prop | 无 | 有 |
| S-TLLR | 无 | 有 |
| TESS | 无 | 无 |

**意义**：TESS 是第一个同时去掉两种 BP 的算法。

**状态**：已冻结。

---

### 6.2 随机矩阵为什么能训练

- 误差不需要精确反向，只需要方向正相关；
- 网络会自适应去配合随机反馈（feedback alignment）；
- 随机投影近似保持向量内积。

**关键**：不是"随机矩阵恰好对"，而是"网络学会了配合随机矩阵"。

**状态**：已冻结。

---

### 6.3 JL 引理和 DFA/TESS 的关系

JL 引理说"$N$ 个点只需 $O(\log N)$ 维"，这里的 $N$ 是**点的个数**，不是向量维度。

- JL：$N$ 个高维点 → $O(\log N)$ 维，保持点间距离；
- TESS：$n^{(l)}$ 维输出 → $C$ 维，因为任务需要 $C$ 维。

**两者目标不同**：JL 是几何保持，TESS 是任务信息保留。TESS 用 $C$ 维是任务决定的，不是 $O(\log N)$。

**状态**：已冻结。

---

## 七、三因子规则

### 7.1 哪些是三因子

| 算法 | 是否三因子 |
|---|---|
| BPTT | 不是 |
| RTRL | 不是 |
| e-prop | 是 |
| OTTT | 形式上 |
| NDOT | 形式上 |
| S-TLLR | 是（明确定义） |
| TESS | 是 |

三因子形式：$\Delta W = \delta \cdot e$

**状态**：已冻结。

---

## 八、信息量化与资源量化

### 8.1 四个量化方向

| 方向 | 内容 |
|---|---|
| 数值量化 | FP32 → INT8/INT4，硬件方向 |
| 信息量化 | 梯度 → 标量（$\cos$, $D_g$, erank, $\Delta L$） |
| 资源量化 | 内存/计算/通信 → 标量 |
| 表示量化 | 逐层探针、CKA、梯度对齐 |

**本文主线**：信息量化 + 资源量化 + 表示量化。数值量化留给硬件方向。

**状态**：已冻结。

---

### 8.2 三个研究方向（表示量化）

**方向一：逐层探针**
$$\text{Acc}^l_{\text{probe}} = \text{Acc}(\text{Probe}(o^l), y)$$

**方向二：逐层 CKA**
$$\text{CKA}(X, Y) = \frac{\|X^\top Y\|_F^2}{\|X^\top X\|_F \cdot \|Y^\top Y\|_F} \in [0, 1]$$

**方向三：逐层梯度对齐**
$$\cos(g^l_{\text{alg}}, g^l_{\text{BPTT}}) = \frac{(g^l_{\text{alg}})^\top g^l_{\text{BPTT}}}{\|g^l_{\text{alg}}\| \cdot \|g^l_{\text{BPTT}}\|}$$

**三者关系**：

- 探针：这层有没有信息？
- CKA：这层信息像不像 BPTT？
- 梯度对齐：这层信息能不能有效优化？

**状态**：已冻结（具体实验设计待做）。

---

### 8.3 Pareto 分析

不合成单一指标 $\eta = \text{Info}/\text{Cost}$，而是画三张散点图：

$$(\text{Memory}, \text{Fidelity})$$
$$(\text{Time}, \text{Fidelity})$$
$$(\text{Communication}, \text{Fidelity})$$

**原因**：Info 的定义不唯一（$\cos$、$1-D_g$、erank 三者意义不同），除以 Cost 没有物理意义。

**状态**：已冻结。

---

## 九、正定矩阵

### 9.1 定义

对称矩阵 $A$ 是正定的，当且仅当对任意非零向量 $x$：

$$x^\top A x > 0$$

**等价条件**：所有特征值为正；所有顺序主子式大于零。

**几何意义**：描述一个开口朝上的碗，原点是全局最小值。

---

### 9.2 在 DFA 收敛性证明中的作用

要求 $WB$ 对称正定。

- 正定保证所有特征值为正；
- $\text{tr}(WB) > 0$；
- DFA 更新方向与真实梯度正相关。

**状态**：已冻结。

---

## 十、待解决的疑问

### 10.1 需要精读原文的

- **疑问 A**：OTTT 论文 Theorem 1 的收敛保证条件（Appendix A）。
- **疑问 B**：NDOT 论文 Appendix A.2 的数值稳定 clamp 具体逻辑。
- **疑问 C**：DFA 原始论文（Lillicrap 2016、Nøkland 2016）的收敛性证明。
- **疑问 D**：TESS 论文中 B 矩阵准正交性的具体证明。

### 10.2 需要自己推导的

- **疑问 E**：LIF 三种复位形式下 OTTT 近似的质量比较。
- **疑问 F**：e-prop 三种反馈选择（symmetric / random / adaptive）的收敛速度比较。
- **疑问 G**：资格迹粒度（每突触 vs 每层）对训练稳定性的影响。

### 10.3 需要实验解决的

- **疑问 H**：逐层探针的准确率曲线是否真能定位信息损失。
- **疑问 I**：逐层 CKA 与梯度对齐是否一致。
- **疑问 J**：信息损失与资源节省的 Pareto 前沿形态。

---

## 十一、关键约定（已冻结）

| 约定 | 内容 |
|---|---|
| LIF 复位形式 | 形式 A：$u_{t+1} = \lambda(u_t - V_{th}s_t) + I$ |
| 时间索引 | $A_t = \partial u^l[t]/\partial u^l[t-1]$，spike Jacobian 用 $D_{\Psi,t-1}$ |
| 层索引 | 用 $W^{l \leftarrow l-1}$ |
| 代理梯度 | fast sigmoid，$\Psi'(x) = 1/(1+\beta|x|)^2$，$\beta = 4$ |
| 更新时机 | 梯度保真度用 Accumulated；在线性能用 Online |
| 参数顺序 | 从浅到深，行优先，与 PyTorch 默认一致 |
| 输出层 | 用 $E_G$；hidden layers 用 local objective |

---

## 十二、下一步

### 阶段 0：搭建实验骨架

**目标**：先搭一套通用实验框架，架子上只挂一个能跑通的东西。

**任务**：

1. 数据与脉冲编码模块（DVS Gesture 为主，CIFAR10-DVS 为辅，$T = 10$）；
2. 统一网络骨架 `SNN(nn.Module)`，支持返回每层 $u, s$ 和固定参数顺序；
3. 统一训练循环 `train_epoch()`，支持 Accumulated 和 Online 两种模式；
4. 指标接口 `evaluate()`，预留探针、CKA、梯度对齐的调用点。

### 阶段 1：挂 BPTT-SG，跑穿上限

**目标**：把 BPTT-SG 作为唯一算法插入骨架，跑出一条完整的收敛曲线。

**任务**：

1. 手写 BPTT-SG 的 forward 和 backward，用形式 A 和 detached-reset；
2. 通过 Gate A / B / C：$A_t^{\text{manual, SG}} \approx A_t^{\text{AD, SG}}$，$B_t^{\text{manual}} \approx B_t^{\text{AD}}$，$g_{\text{BPTT}}^{\text{manual}} \approx g_{\text{BPTT}}^{\text{AD}}$，相对误差 $< 10^{-5}$；
3. 跑通 DVS Gesture，记录每层的 $u, s$ 和梯度。

### 阶段 2：焊上探针、CKA、梯度对齐的钩子

**目标**：在写其他算法前，先把测量工具装好。

**任务**：

1. 逐层探针模块：每层挂独立线性探针，不参与主网络训练；
2. 逐层 CKA 模块：收集每层表示矩阵，计算与 BPTT 的 CKA；
3. 逐层梯度对齐模块：收集每层梯度，计算与 BPTT 的余弦相似度；
4. 在 BPTT 自己身上验证工具本身：$\text{CKA} = 1$，$\cos = 1$。

### 阶段 3：挂载 OTTT / NDOT / e-prop

**目标**：同一框架下逐个挂载其他算法。

**任务**：

1. OTTT：实现 $\hat{a}^{l-1}[t]$ 前向递推与瞬时梯度，验证 trace 递推与闭式一致；
2. NDOT：实现动态系数 $e^l[t]$ 与动态迹递推，注意 $t=1$ 特例；
3. e-prop：实现 ideal factorization 与 eligibility 局部递推，验证 $\frac{dE}{dW_{ji}} = \sum_t L_j^t e_{ji}^t$。

### 阶段 4：接入 S-TLLR / TESS

**目标**：外部验证，优先使用作者公开代码。

**任务**：

1. 下载 S-TLLR 和 TESS 作者代码，在统一数据下复现论文报告的数字；
2. 逐层测量三个指标，确保测量口径一致。

### 阶段 5：逐层评估实验

**目标**：回答核心研究问题。

**任务**：

1. 逐层画三张图：探针准确率、CKA、梯度对齐；
2. 归因分析：temporal 近似（OTTT vs NDOT）、spatial 局部化（S-TLLR vs TESS）、目标替换（BPTT vs TESS）；
3. Pareto 分析：Memory-Fidelity、Time-Fidelity、Communication-Fidelity 三张散点图。

### 阶段 6：与硬件方向对接

**目标**：把逐层信息分布整理成硬件可参考的规格。

**任务**：

1. 状态清单：每个算法需要存储哪些状态、状态生命周期；
2. 通信清单：每层通信需求、哪些可以本地化；
3. 更新清单：更新频率、更新模式、权重写入次数；
4. 与硬件同学讨论，定期反馈可行性。

### 阶段之间的依赖

阶段 0 → 阶段 1 → 阶段 2 → 阶段 3 → 阶段 4 → 阶段 5 → 阶段 6。

**关键原则**：先搭架子，架子上只挂一个能跑穿的东西（BPTT-SG），然后再逐个挂载其他算法。探针和 CKA 的钩子必须在阶段 2 就焊上。

---

*文档版本：v1.1*
*最后更新：2026-10-05*
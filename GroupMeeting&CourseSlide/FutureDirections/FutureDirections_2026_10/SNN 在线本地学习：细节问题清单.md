# SNN 在线/本地学习：细节问题清单（2026-10）

> 本文档记录读论文、推导公式、设计实验过程中出现的细节问题，分为「已解决（含详细推导）」和「待解决」两大类。
> 上一版见 `FutureDirections_2026_09\SNN 在线本地学习：细节问题清单.md`。

---

## 一、已解决（含详细推导）

### 1. 符号与约定

#### 1.1 LIF 复位形式

三种常见写法：

- **A**：$u_{t+1} = \lambda(u_t - V_{th}s_t) + I$
- **B**：$u_{t+1} = \lambda u_t - V_{th}s_t + I$
- **C**：$u_{t+1} = \lambda u_t(1-s_t) + I$

对应的时间依赖矩阵 $\epsilon^l[i]$（对 $u_t$ 求偏导）：

| 形式 | $\epsilon^l[i]$ |
|---|---|
| A | $\lambda I - \lambda V_{th} D_{\Psi,i}$ |
| B | $\lambda I - V_{th} D_{\Psi,i}$ |
| C | $\lambda(1-s)I - \lambda u D_{\Psi,i}$ |

**为什么会影响推导**：三种形式的区别在复位项的位置。形式 A 的 reset 被 $\lambda$ 一起衰减，形式 B 的 reset 不衰减，形式 C 是乘性复位。这直接改变 $\epsilon^l[i]$ 里 reset 项的系数。

**推导（以形式 A 为例）**：
$$u_{t+1} = \lambda u_t - \lambda V_{th} s_t + I, \quad s_t = H(u_t - V_{th})$$
对 $u_t$ 求偏导，注意 $s_t$ 也依赖 $u_t$（通过 $H$），用代理梯度 $D_{\Psi,i}=\operatorname{diag}[\Psi'(u_j - V_{th})]_j$ 替代 $\partial s_t/\partial u_t$：
$$\epsilon^l[i] = \frac{\partial u_{i+1}}{\partial u_i} = \lambda I - \lambda V_{th} D_{\Psi,i}$$

**本文约定**：统一用形式 A。理由是与 OTTT / NDOT / e-prop 的符号约定最接近，跨算法比较不用换符号。

#### 1.2 全导 vs 偏导

链式法则中：
$$\frac{dL}{dW_{ji}^l} = \sum_t \frac{\partial L}{\partial s_j^l[t]} \cdot \frac{\partial s_j^l[t]}{\partial u_j^l[t]} \cdot \frac{du_j^l[t]}{dW_{ji}^l}$$

**为什么最后一项必须是全导**：$W_{ji}^l$ 通过两条路径影响 $u_j^l[t]$：

- 直接路径：$t$ 时刻的输入 $W_{ji}^l s_i^{l-1}[t]$ 进入 $u_j^l[t]$；
- 间接路径：$W_{ji}^l$ 影响 $u_j^l[t-1]$，再经过状态转移到 $u_j^l[t]$。

如果写偏导 $\partial u_j^l[t]/\partial W_{ji}^l$，右边直接等于 $s_i^{l-1}[t]$，**丢掉所有历史路径**，这就不是 BPTT。

**全导展开**：
$$\frac{du_j^l[t]}{dW_{ji}^l} = \sum_{\tau \le t} \frac{\partial u_j^l[t]}{\partial u_j^l[\tau]} \cdot \frac{\partial u_j^l[\tau]}{\partial W_{ji}^l}$$

注意展开后的 $\partial u_j^l[\tau]/\partial W_{ji}^l$ 是偏导，因为 $u_j^l[\tau]$ 对权重的直接依赖只通过 $\tau$ 时刻那一项。

#### 1.3 时间索引

$A_t \equiv \partial u^l[t]/\partial u^l[t-1]$，因此 $\epsilon^l[i] = \partial u^l[i+1]/\partial u^l[i]$ 内部所有 spike Jacobian 项用 $D_{\Psi,i}$（下标 $i$，不用 $D_{\Psi,t}$）。

**为什么**：$\epsilon^l[i]$ 的下标是 $i$，表示从时间步 $i$ 到 $i+1$ 的转移，所以 $\epsilon^l[i]$ 里出现的 $s^l[i]$ 对 $u^l[i]$ 求导，得到 $D_{\Psi,i}$。

#### 1.4 层索引映射

不同论文对 $W^l$ 定义不同：

- OTTT 原文：$W^l$ 表示 $l \to l+1$；
- NDOT 原文：$W^l$ 表示 $l-1 \to l$。

本文统一用 $W^{l \leftarrow l-1}$ 表示从 $l-1$ 到 $l$ 的权重。

### 2. 资格迹

#### 2.1 资格迹的维护粒度

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

#### 2.2 资格迹如何适配不同层的维度

外积自动适配：
$$\nabla_{W^{l \leftarrow l-1}} L[t] = g_{u^l}[t] \otimes \hat{a}^{l-1}[t]^\top$$

维度：

- $g_{u^l}[t] \in \mathbb{R}^{n^{(l)}}$（突触后维度）
- $\hat{a}^{l-1}[t] \in \mathbb{R}^{n^{(l-1)}}$（突触前维度）
- 外积：$\mathbb{R}^{n^{(l)} \times n^{(l-1)}}$，正好等于 $W^{l \leftarrow l-1}$ 的形状。

**不同层用自己的资格迹**：层 0 → 层 1 用 $\hat{a}^0$，层 1 → 层 2 用 $\hat{a}^1$。**不跨层共享**。

#### 2.3 $D_{\Psi,t}$ 是矩阵还是向量

$$D_{\Psi,t} = \operatorname{diag}[\Psi'(u_j^l[t] - V_{th})]_j \in \mathbb{R}^{n^{(l)} \times n^{(l)}}$$

**是矩阵，不是向量**。

**为什么**：它要参与矩阵运算 $\epsilon^l[i] = \lambda I - \lambda V_{th} D_{\Psi,i}$。$\lambda I$ 是 $n \times n$ 矩阵，$D_{\Psi,i}$ 也必须是 $n \times n$ 矩阵才能相减。

**物理含义**：第 $j$ 个对角元 $\Psi'(u_j^l[t] - V_{th})$ 衡量神经元 $j$ 对膜电位的敏感度。膜电位接近阈值时最大，远离阈值时快速衰减。

#### 2.4 TESS 中 $\odot$ 和 $\otimes$ 的区别

- $\odot$：逐元素乘法（Hadamard product），**同维度**输入，输出维度不变；
- $\otimes$：外积，**不同维度**输入，输出是两个维度的乘积。

TESS 的权重更新：
$$\Delta W^{(l)}[t] = \big(m^{(l)}[t] \odot \alpha_{\text{pre}} \Psi(u^{(l)}[t])\big) \otimes q^{(l)}[t] + \big(m^{(l)}[t] \odot \alpha_{\text{post}} h^{(l)}[t]\big) \otimes o^{(l-1)}[t]$$

**流程**：

1. 先用 $\odot$ 让每个神经元的 learning signal 调制它自己的 activation（仍然是向量，维度 $n^{(l)}$）；
2. 再用 $\otimes$ 把"突触后调制后的 activation"与"突触前 trace"组合，生成突触级权重更新（矩阵，维度 $n^{(l)} \times n^{(l-1)}$）。

**为什么要两个符号**：$\odot$ 用于"调制"（维度不变），$\otimes$ 用于"组合"（生成矩阵）。

#### 2.5 恒等式 $(m \odot a) \otimes b = m \odot (a \otimes b)$

这是 TESS 内存从 $O(n^2)$ 降到 $O(n)$ 的数学基础。

**证明**：

- 左边第 $(i,j)$ 元素：$(m_i \cdot a_i) \cdot b_j$
- 右边第 $(i,j)$ 元素：$m_i \cdot (a_i \cdot b_j)$
- 两者相等。

**意义**：TESS 不需要显式存储 $n \times n$ 的突触级资格迹 $e$，只需要存两个神经元级向量 $q, h$，更新权重时用外积即时构造 $e$。

### 3. 更新时机

#### 3.1 两种更新模式

**Accumulated 模式**：
$$W \leftarrow W - \eta \sum_{t=1}^{T} \nabla_W L[t]$$
序列内 $W$ 不变，末尾更新一次。

**Online 模式**：
$$W \leftarrow W - \eta \nabla_W L[t]$$
每个时间步覆盖式更新。

#### 3.2 各算法的更新模式

| 算法 | 更新模式 | 说明 |
|---|---|---|
| BPTT | 天然 Accumulated | 序列末尾更新一次 |
| e-prop | Ideal 累积 / Online 每步 | 取决于 learning signal 用 ideal 还是近似 |
| OTTT | O 模式每步 / A 模式累积 | 两种都支持 |
| NDOT | O 模式每步 / A 模式累积 | 两种都支持 |
| S-TLLR | 每步更新 | 但只有最后几步有学习信号 |
| TESS | 累积末尾更新 | Algorithm 1 明确 |

**梯度保真度比较必须用 Accumulated 模式。**

#### 3.3 权重 $W$ 的存储

**两种模式都只存一份 $W$。**

- Accumulated：$W$ 序列内不变，末尾更新一次；
- Online：$W$ 每步覆盖，历史版本不保留。

内存里始终只有一份 $W$ 和（如果需要）一份梯度累积缓冲区。**不需要保存历史 $W$**：前向只依赖当前 $W$，梯度只在当前 $W$ 上算，更新是覆盖式的。

#### 3.4 为什么 Online 模式不能直接和 BPTT 比较梯度

Online 模式下：

- $t=1$ 的梯度在 $W_0$ 上算；
- $t=2$ 的梯度在 $W_1 = W_0 - \eta g_1$ 上算；
- $t=3$ 的梯度在 $W_2$ 上算；……

$\sum_t g_t(W_{t-1})$ 不是同一个 $W$ 下的梯度总和，和 BPTT 的 $g$ 不在同一个评估点。

**后果**：如果强行比较，混淆了两部分——算法近似和评估点漂移。**无法归因**。

**Accumulated 模式消除混淆**：整段序列用同一个 $W_0$，$\sum_t g_t(W_0)$ 和 BPTT 的 $g$ 可以直接比较。

### 4. 算法理解

#### 4.1 e-prop、OTTT、NDOT 的关系

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

#### 4.2 e-prop 的 eligibility 是否精确

**eligibility 是精确的**。它是 chain-rule rearrangement 后的局部项，可以由突触和 postsynaptic neuron 的局部 forward variables 计算。

**近似发生在 learning signal**：$L_j^t = dE/dz_j^t$ 需要未来信息，在线时用 $\hat{L}_j^t$ 近似。

#### 4.3 S-TLLR 的 non-causal 项

"non-causal" 是 STDP timing 意义上的 non-causal——当前 pre spike 与过去 post activity 的相关性。**不访问未来**，仍是 online。

**公式**：
$$e_{ij}[t] = \alpha_{\text{pre}} \Psi(u_i[t]) \sum_{t'=0}^{t} \lambda_{\text{pre}}^{t-t'} x_j[t'] + \alpha_{\text{post}} x_j[t] \sum_{t'=0}^{t-1} \lambda_{\text{post}}^{t-t'} \Psi(u_i[t'])$$

第一项（因果）：当前 post × 过去 pre 迹。第二项（非因果）：当前 pre × 过去 post 迹（减去当前）。

#### 4.4 S-TLLR 的 $\Psi$ 和 BPTT 的 $\Psi'_{\text{SG}}$

两者是不同对象：

- $\Psi$：secondary activation，作用于膜电位，生成连续突触后活动。用于资格迹的平滑；
- $\Psi'_{\text{SG}}$：spike surrogate derivative，替代不可导的 $\partial s/\partial u$。用于反向传播。

$$\Psi(u) \neq \Psi'_{\text{SG}}(u)$$

#### 4.5 TESS 和 S-TLLR 的核心区别

- S-TLLR：时间局部（STDP 迹），但空间仍需要 BP/DFA；
- TESS：时间局部（STDP 迹）+ 空间局部（本地 LSG）。

TESS 是 S-TLLR 的"空间也局部化"版本。**关键**：S-TLLR 的 $\delta_i[t]$ 由跨层 BP/DFA 得到，TESS 的 $m^{(l)}[t]$ 完全在层内生成。

#### 4.6 TESS 的 B 矩阵为什么能替代跨层反传

来自 DFA 的思想：

- 用固定随机矩阵 $B^{(l)}$ 替代真实反向权重 $W^{(l+1)\top}$；
- 误差不需要精确反向，只需要方向正相关；
- 训练过程中前向权重会自适应去配合随机反馈（feedback alignment）。

**核心直觉**：不是"随机矩阵恰好对"，而是"网络会学会配合随机矩阵"。

#### 4.7 TESS 的 B 矩阵是固定的还是训练的

**固定的**。好处：硬件友好（不需要存储梯度）；实验表明效果足够；训练 $B$ 会引入额外的梯度路径，复杂度增加。

### 5. 时空解耦

#### 5.1 解耦的本质

$$\nabla_W L = \underbrace{(\text{空间因子})}_{\text{独立算}} \times \underbrace{(\text{时间因子})}_{\text{独立算}}$$

两个因子独立计算，最后乘积组合。**不是"互相不影响"**，而是"分别计算，最后组合"。

#### 5.2 解耦需要什么条件

**数学上的精确解耦无条件**——它是链式法则重排的恒等式。

**在线解耦需要四个条件**：

1. **隐藏状态局部**：$h_j^t$ 只依赖 $h_j^{t-1}$ 和外部输入；
2. **权重影响局部**：$W_{ji}$ 只直接影响 $h_j$；
3. **马尔可夫性**：$h_j^t$ 只依赖 $h_j^{t-1}$；
4. **代理梯度可用**：用光滑函数替代 $\partial s/\partial u$。

LIF / ALIF 满足前三条。**如果违反**（如跨神经元耦合动力学），资格迹维度从 $O(n)$ 回到 $O(n^2)$。

#### 5.3 解耦的代价

- 资格迹 $e_{ji}^t$：**精确保留**（可以前向递推）；
- 学习信号 $L_j^t$：**必须近似**（需要未来信息）。

$$\boxed{\text{解耦 = 时间因子精确保留，空间因子必须近似}}$$

**这就是为什么不同算法的差异集中在"怎么近似 $L_j^t$"**：OTTT/NDOT 用瞬时信号，e-prop 用反馈权重，TESS 用本地 LSG。

### 6. 空间局部化

#### 6.1 Spatial BP 是什么

沿"层"方向反向传播误差，与 Temporal BP（沿时间）相对。

| 算法 | Temporal BP | Spatial BP |
|---|---|---|
| BPTT | 有 | 有 |
| OTTT / NDOT | 无 | 有 |
| e-prop | 无 | 有 |
| S-TLLR | 无 | 有 |
| TESS | 无 | 无 |

**意义**：TESS 是第一个同时去掉两种 BP 的算法。

#### 6.2 随机矩阵为什么能训练

- 误差不需要精确反向，只需要方向正相关；
- 网络会自适应去配合随机反馈（feedback alignment）；
- 随机投影近似保持向量内积。

**关键**：不是"随机矩阵恰好对"，而是"网络学会了配合随机矩阵"。

#### 6.3 JL 引理和 DFA/TESS 的关系

JL 引理说"$N$ 个点只需 $O(\log N)$ 维"，这里的 $N$ 是**点的个数**，不是向量维度。

- JL：$N$ 个高维点 → $O(\log N)$ 维，保持点间距离；
- TESS：$n^{(l)}$ 维输出 → $C$ 维，因为任务需要 $C$ 维。

**两者目标不同**：JL 是几何保持，TESS 是任务信息保留。TESS 用 $C$ 维是任务决定的，不是 $O(\log N)$。

### 7. 三因子规则

#### 7.1 哪些是三因子

| 算法 | 是否三因子 |
|---|---|
| BPTT | 不是 |
| RTRL | 不是 |
| e-prop | 是 |
| OTTT | 形式上 |
| NDOT | 形式上 |
| S-TLLR | 是（明确定义） |
| TESS | 是 |

三因子形式：$\Delta W = \delta \cdot e$（学习信号 $\times$ 资格迹）。

### 8. 信息量化与资源量化

#### 8.1 四个量化方向

| 方向 | 内容 |
|---|---|
| 数值量化 | FP32 → INT8/INT4，硬件方向 |
| 信息量化 | 梯度 → 标量（$\cos$, $D_g$, erank, $\Delta L$） |
| 资源量化 | 内存/计算/通信 → 标量 |
| 表示量化 | 逐层探针、CKA、梯度对齐 |

**本文主线**：信息量化 + 资源量化 + 表示量化。数值量化留给硬件方向。

#### 8.2 三个研究方向（表示量化）

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

#### 8.3 Pareto 分析

不合成单一指标 $\eta = \text{Info}/\text{Cost}$，而是画三张散点图：

$$(\text{Memory}, \text{Fidelity}), \quad (\text{Time}, \text{Fidelity}), \quad (\text{Communication}, \text{Fidelity})$$

**原因**：Info 的定义不唯一（$\cos$、$1-D_g$、erank 三者意义不同），除以 Cost 没有物理意义。

### 9. 正定矩阵（详细）

#### 9.1 定义

对称矩阵 $A$ 是**正定**的，当且仅当对任意非零向量 $x$：
$$x^\top A x > 0$$

**等价条件**（三者等价）：

1. 所有特征值为正：$\lambda_i(A) > 0, \forall i$；
2. 所有顺序主子式（leading principal minors）大于零；
3. 存在满秩矩阵 $P$ 使 $A = P^\top P$（Cholesky 分解存在）。

**几何意义**：二次型 $f(x) = x^\top A x$ 描述一个"开口朝上"的碗（凸抛物面），原点是全局最小值。正定保证沿任意方向的曲率都为正，梯度下降不会陷入鞍点/最大值。

**谱分解视角**：$A = Q\Lambda Q^\top$，其中 $Q$ 正交、$\Lambda$ 为特征值对角阵。正定 $\iff \Lambda$ 对角元全正。

#### 9.2 在 DFA 收敛性证明中的作用

DFA（Direct Feedback Alignment）用固定随机矩阵 $B$ 替代真实反向权重 $W^\top$。其收敛性证明的核心是要求**前向权重 $W$ 与反馈权重 $B$ 的乘积 $WB$ 对称正定**：

- **正定**保证 $WB$ 的所有特征值为正；
- 于是 $\operatorname{tr}(WB) = \sum_i \lambda_i(WB) > 0$；
- 这保证 DFA 的更新方向 $-\nabla^{\text{DFA}}$ 与真实梯度 $-\nabla L$ 的内积为正，即**DFA 更新方向与真实梯度正相关**，是有效的下降方向。

**为什么是 $WB$ 而不是 $W$ 单独**：真实 BP 的误差传播用 $W^\top$，DFA 用 $B$。前向传播把 $W$ 叠上去，所以有效反馈映射是 $WB$。当 $WB$ 对称正定时，$WB$ 与单位阵"方向一致"，DFA 近似退化为真实 BP。

**注意**：$WB$ 初始（$B$ 随机时）未必正定，但训练中 $W$ 会自适应（feedback alignment），使 $WB$ 逐渐趋于对称正定。这正是 DFA"网络学会配合随机反馈"的数学刻画。严格证明需精读 DFA 原文（Lillicrap 2016、Nøkland 2016），本文档仅给出结论（见"待解决"疑问 C）。

### 10. 收敛性证明与数值细节（本次阅读论文后新增）

> 本节回答原"待解决"中能通过阅读 Algorithm 论文内容或自行推导解决的问题。

#### 10.1 OTTT 的 Theorem 1（疑问 A）

**来源**：OTTT 论文（NeurIPS 2022）正文 §4 与 Appendix A.3。

**Theorem 1 的完整条件**（三个）：

1. **Assumption 1 成立**：$\forall l = 1,\dots,N,\ t=1,\dots,T$，surrogate derivative 构成的对角矩阵满足某种有界/正则条件（具体形式见原文 Appendix A.3，转换文本中公式排版破碎，此处给定性描述——本质是"surrogate 导数有界，且与最终状态 Jacobian 相关"）。
2. **$V_{th}=1$**：把阈值归一化。
3. **误差 $\epsilon^l[t] = a^l[t] - a^l[T]$ 足够小**：weighted firing rate $a^l[t]$ 逐时刻收敛到最终值 $a^l[T]$，误差满足某个不等式（$\sum_t \hat g_{u^{l+1}}[t]\epsilon^l[t]^\top$ 小于某个界）。

**结论**：在上述条件下，
$$\langle \nabla_{W^l} L,\ (\nabla_{W^l} L_{\text{sr}})_{\text{sr}} \rangle > 0,$$
即 OTTT 梯度与基于 spike representation 的梯度**内积为正**，OTTT 梯度是优化问题（以 spike representation 形式化）的**下降方向**。

**Theorem 2** 是循环网络（recurrent）的对应版本：同样在 Assumption 1、$V_{th}=1$、firing rate 收敛误差小的条件下，证明 OTTT 梯度是下降方向（附录 A.4）。

**直觉**：OTTT 用逐时刻瞬时 trace $\hat a^l[t]$ 替代 spike representation 方法用最终 $a^l[T]$ 的做法；当 firing rate 逐时刻收敛、误差小时，两者梯度方向一致。

#### 10.2 NDOT 的数值稳定 clamp（疑问 B）

**来源**：NDOT 论文（ICML 2024）Appendix A.2。

**背景**：NDOT 的动态系数为
$$e^{l-1}[t] = \frac{u^{l-1}[t] - V_{th}s^{l-1}[t]}{u^{l-1}[t-1] - V_{th}s^{l-1}[t-1]},$$
分母 $u^{l-1}[t-1] - V_{th}s^{l-1}[t-1]$ 可能为零（如初始状态、或膜电位恰等于 $V_{th}s$ 时）。

**处理逻辑**（论文原文）：
> 当 $e^{l-1}[t]$ 的分母 $u^{l-1}[t-1] - V_{th}s^{l-1}[t-1]$ 等于零时，先判断 $u^{l-1}[t-1]$ 的符号，然后用 clamping 函数把值限制在 $[-\lambda, \lambda]$ 范围内。

即：只对分母为零的退化情况生效，不是一般定义；clamp 的上下界是 $[-\lambda, \lambda]$。

#### 10.3 TESS 的 B 矩阵准正交性（疑问 D）

**来源**：TESS 论文（arXiv:2502.01837v1）§（LSG 设计）。

**结论**：$B^{(l)}$ 是**固定的二值矩阵**（尺寸 $C \times n^{(l)}$），**每一列对应一个 square wave（方波）函数**。

**准正交性的来源**：不是通过 Gram–Schmidt 正交化，而是通过给**不同类别分配不同的空间频率**（distinct spatial frequencies）的方波函数。不同频率的方波函数天然近似正交，因此 $B^{(l)}$ 的列之间 quasi-orthogonal（准正交），**最小化不同类别投影之间的干扰**。

**其他优点**：
- 同步层内神经元活动（同一频率的神经元同步激活）；
- 让任务信息分散到整层神经元；
- 方波二值，硬件实现高效（矩阵乘法成本低）。

**注意**：准正交是"近似"而非严格正交（方波函数不完备正交），论文用的是"quasi-orthogonal"这个措辞。

#### 10.4 LIF 三种复位形式下 OTTT 的近似质量（疑问 E，自行推导）

三种形式的时间依赖矩阵：

| 形式 | $\epsilon^l[i]$ | OTTT 丢弃后 |
|---|---|---|
| A | $\lambda I - \lambda V_{th} D_{\Psi,i}$ | $\lambda I$ |
| B | $\lambda I - V_{th} D_{\Psi,i}$ | $\lambda I$ |
| C | $\lambda(1-s)I - \lambda u D_{\Psi,i}$ | $\approx \lambda I$ |

**推导**：OTTT 的核心近似是"丢弃复位项（含 $D_{\Psi}$ 的项）"。三种形式的被丢弃残差：

- **A**：丢弃 $-\lambda V_{th} D_{\Psi,i}$，残差被 $\lambda < 1$ 衰减，**最小**；
- **B**：丢弃 $-V_{th} D_{\Psi,i}$，残差**无 $\lambda$ 衰减**，比 A 大（当 $\lambda<1$ 时 $|-\lambda V_{th}D|<|-V_{th}D|$）；
- **C**：丢弃 $- \lambda u D_{\Psi,i}$，残差含 $u$（膜电位可大于 $V_{th}$），且泄漏项 $\lambda(1-s)$ 本身还含 $s$ 因子，丢弃后引入的误差**最复杂、最大**。

**结论**：OTTT 近似质量 **A > B > C**。形式 A 的复位项被 $\lambda$ 一起衰减，丢弃它造成的误差最小。这也从理论上支持了本文"统一用形式 A"的约定（见 1.1）。

#### 10.5 资格迹粒度对训练的影响（疑问 G，理论部分）

| 粒度 | 算法 | 内存 | 信息 |
|---|---|---|---|
| 每突触 | e-prop | $O(Ln^2)$ | 精确保留每个突触 $(i,j)$ 的历史 |
| 每层 | OTTT / NDOT | $O(Ln)$ | 整层共享一个 trace，丢失突触间差异 |
| 神经元级外积 | S-TLLR / TESS | $O(Ln)$ | 用两个神经元级向量即时构造突触级 |

**理论影响**：粒度越粗，丢失的突触间差异信息越多。每突触粒度精确保留"这个突触的 pre 脉冲历史"，每层粒度则把所有以该层为输入的突触混在同一个 trace 里——当不同突触的 pre 活动差异大时，共享 trace 会引入**梯度方向偏差**，可能影响训练稳定性与收敛质量。

**定量影响需实验**（见"待解决"疑问 G 的实验部分）。

---

## 二、待解决

### 1. 需精读原文（Algorithm 内无对应原文）

- **疑问 C**：DFA 原始论文（Lillicrap 2016、Nøkland 2016）的收敛性证明。已得到结论（$WB$ 对称正定，见 9.2），但完整证明需精读原文，Algorithm 文件夹内没有 DFA 论文。

### 2. 需实验解决（概念已知，定量比较需实验）

- **疑问 F**：e-prop 三种反馈选择（symmetric / random / adaptive）的收敛速度比较。概念已知（对称 = 转置权重，随机 = 固定随机矩阵，自适应 = 学习反馈权重），但"谁收敛更快"是实验问题，Algorithm 内只有 e-prop 的 pdf（无转换文本），无法从论文直接给出定量结论。
- **疑问 G（实验部分）**：资格迹粒度（每突触 vs 每层）对训练稳定性的定量影响（理论部分见 10.5）。
- **疑问 H**：逐层探针的准确率曲线是否真能定位信息损失。
- **疑问 I**：逐层 CKA 与梯度对齐是否一致。
- **疑问 J**：信息损失与资源节省的 Pareto 前沿形态。

---

## 三、关键约定汇总表（已冻结）

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

*文档版本：v3.0（2026-10，详细版 + 已解决/待解决二分 + 收敛性解答）*




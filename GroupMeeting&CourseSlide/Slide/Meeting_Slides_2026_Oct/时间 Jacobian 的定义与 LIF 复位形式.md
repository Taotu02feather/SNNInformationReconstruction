# 时间 Jacobian 的定义与 LIF 复位形式 A/B/C

## 时间 Jacobian 的定义

时间 Jacobian 描述相邻两个时间步膜电位之间的依赖关系，定义为

$$
\epsilon[t-1] \;=\; \frac{\partial u[t]}{\partial u[t-1]}.
$$

其中 $u[t]$ 是第 $t$ 步的膜电位向量。$\epsilon[t-1]$ 是矩阵，刻画 $u[t]$ 对 $u[t-1]$ 的一阶依赖。

在 BPTT 的全导数展开中，$\epsilon[t-1]$ 作为时间反传的连乘因子出现：

$$
\frac{du[t]}{dW}
=
s[t]
+
\sum_{k<t}
\left(\prod_{i=k}^{t-1}\epsilon[i]\right) s[k].
$$

## 形式 A

$$
u[t] = \lambda\left(u[t-1] - V_{th}s[t-1]\right) + W s[t] + b
$$

复位项 $V_{th}s[t-1]$ 被 $\lambda$ 一起衰减。

时间 Jacobian：

$$
\epsilon[t-1] = \lambda I - \lambda V_{th} D_{\Psi,t-1}.
$$

## 形式 B

$$
u[t] = \lambda u[t-1] - V_{th}s[t-1] + W s[t] + b
$$

复位项 $V_{th}s[t-1]$ 不被 $\lambda$ 衰减。

时间 Jacobian：

$$
\epsilon[t-1] = \lambda I - V_{th} D_{\Psi,t-1}.
$$

## 形式 C

$$
u[t] = \lambda u[t-1]\left(1 - s[t-1]\right) + W s[t] + b
$$

复位为乘性：$u[t-1]$ 乘 $(1-s[t-1])$。

时间 Jacobian：

$$
\epsilon[t-1] = \lambda\left(1 - s[t-1]\right) I - \lambda u[t-1] D_{\Psi,t-1}.
$$

## 三者的区别

| 形式 | 复位方式 | 时间 Jacobian $\epsilon[t-1]$ |
|---|---|---|
| A | 减性，被 $\lambda$ 衰减 | $\lambda I - \lambda V_{th} D_{\Psi,t-1}$ |
| B | 减性，不被 $\lambda$ 衰减 | $\lambda I - V_{th} D_{\Psi,t-1}$ |
| C | 乘性 | $\lambda(1-s[t-1]) I - \lambda u[t-1] D_{\Psi,t-1}$ |

- A 与 B 的区别：复位项系数是 $\lambda V_{th}$ 还是 $V_{th}$；
- A 与 C 的区别：复位是减性还是乘性，C 的 Jacobian 第一项含 $(1-s[t-1])$、第二项含 $u[t-1]$，都随时间动态变化。

## 保留与舍去复位路径项

三种形式都含复位路径项（A 的 $-\lambda V_{th}D_{\Psi,t-1}$、B 的 $-V_{th}D_{\Psi,t-1}$、C 的 $-\lambda u[t-1]D_{\Psi,t-1}$）。

- **完整 BPTT**：保留该项，$\epsilon$ 如上表；
- **OTTT / detached reset**：舍去该项，A、B 的 $\epsilon$ 退化为 $\lambda I$，C 退化为 $\lambda(1-s[t-1])I$。

本文统一用形式 A。

## 时间 Jacobian 是本地学习的核心简化对象

BPTT 的时间反传依赖连乘

$$
\prod_{i=k}^{t-1}\epsilon[i],
$$

这要求网络存储所有时间步的 $u, s$，内存与时间步数成正比。本地学习算法要摆脱的正是这条全局时间链，而它们采取的途径都是对同一个 $\epsilon[t-1]$ 做简化或替换：

- **OTTT**：$\epsilon[t-1] \approx \lambda I$，丢掉复位路径项 $-\lambda V_{th}D_{\Psi,t-1}$，时间依赖退化为固定衰减；
- **NDOT**：保留逐元素结构，用动态系数 $e^l[t]$ 替代固定 $\lambda$，比 OTTT 更接近完整 $\epsilon$，但仍是对角近似；
- **e-prop**：用局部资格迹近似 $\epsilon$ 连乘的贡献，不显式存储 Jacobian；
- **S-TLLR / TESS**：换用 STDP 风格的资格迹机制替代显式时间反传。

四篇论文方法各异，本质都是在用不同方式近似同一个时间 Jacobian。

## 与本研究的关系

本研究提出的逐层信息保留与损失定位，在时间维度上即对比不同算法对 $\epsilon[t-1]$ 的近似与完整 BPTT 的偏差：

- 完整 BPTT：$\epsilon[t-1] = \lambda I - \lambda V_{th}D_{\Psi,t-1}$；
- OTTT：$\epsilon[t-1] = \lambda I$；
- NDOT：动态对角近似；
- 每层累计的偏差，即本地学习相对 BPTT 在时间维度的信息损失。

阶段 1 实现的 detached（$\lambda I$）与 full（$\lambda I - \lambda V_{th}D_{\Psi}$）对照，是这一分析在最小尺度上的预演：先把丢掉复位路径项这一项的代价单独拎出量化，作为后续对比 OTTT、NDOT 的基线。
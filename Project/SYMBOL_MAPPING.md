# SYMBOL_MAPPING —— 论文记法 ↔ 代码记法 对照备忘录

> 本文档专门写清「论文/tex 记法」与「代码记法」两套符号的对应关系，作为唯一允许两套符号同时出现的对照锚点。
> 其余所有文档（论文、tex、md）只用论文记法；所有代码只用代码记法；二者对照只在本文档出现。

---

## 一、差异来源（为什么差 1）

两套记法的层索引基准不同，**差 1 是记法差异，不是错误**：

- **论文记法**（见 `FutureDirections_2026_10/SNN 在线本地学习：细节问题清单.md` §1.4）：权重上标是**目标层（突触后层）**，**1-based**。
  记 $W^{l\leftarrow l-1}$ 表示「从第 $l-1$ 层 → 第 $l$ 层」，方向 $l-1 \to l$。
- **代码记法**（见 `network/mlp_snn.py`）：`weights[l]` 的下标是**突触前层**，**0-based**。
  `weights[l]` 连接「第 $l$ 层 → 第 $l+1$ 层」，形状 `(n_{l+1}, n_l)`。

**核心对应关系**（记 $L = \text{len(layer\_sizes)} - 1$ 为权重矩阵个数）：

> - 论文层 $l$（$l = 1\dots L$，即非输入层） ↔ 代码索引 $l-1$（$0\dots L-1$）。
> - 论文 $W^{l\leftarrow l-1}$（$l\ge1$） ↔ 代码 `weights[l-1]`。
> - 代码 `weights[l]` ↔ 论文 $W^{(l+1)\leftarrow l}$。

**具体例子**（`layer_sizes = [8, 16, 3]`，即 $L=2$）：

| 代码 | 形状 | 连接方向 | 论文记法 |
|---|---|---|---|
| `weights[0]` | `(16, 8)` | 输入(8) → 隐藏(16) | $W^{1\leftarrow 0}$ |
| `weights[1]` | `(3, 16)` | 隐藏(16) → 输出(3) | $W^{2\leftarrow 1}$ |
| `layer_u[0]` | `(T, B, 16)` | 隐藏层膜电位 | $u^1[t]$ |
| `layer_u[1]` | `(T, B, 3)` | 输出层膜电位 | $u^2[t]$ |

---

## 二、完整对照表

下表中「论文 $l$」表示非输入层编号 $l = 1\dots L$，与之对应的「代码 $l-1$」为 0-based 索引 $0\dots L-1$。

| 量 | 论文记法 | 代码记法 | 对应关系 |
|---|---|---|---|
| 层间权重 | $W^{l\leftarrow l-1}$（$l-1\to l$） | `weights[l-1]` | `weights[l]` = $W^{(l+1)\leftarrow l}$ |
| 层间偏置 | $b^{l}$（第 $l$ 层偏置） | `biases[l-1]` | `biases[l]` = $b^{l+1}$ |
| 膜电位 | $u^l[t]$ | `layer_u[l-1][t]` | 第 $l$ 层膜电位序列 |
| 脉冲 | $s^l[t]$ | `layer_s[l-1][t]` | 第 $l$ 层脉冲序列 |
| 输入层活动 | $s^0[t] = x[t]$ | `x[t]` | 输入层无膜电位，直接外部输入 |
| 反向误差 | $\delta^l[t]=\partial L/\partial u^l[t]$ | `delta[l-1][t]` | 第 $l$ 层对 $u$ 的梯度 |
| 权重梯度 | $dW^{l\leftarrow l-1}$ | `dW[l-1]` | 对应 `weights[l-1]` |
| 偏置梯度 | $db^{l}$ | `db[l-1]` | 对应 `biases[l-1]` |
| 时间 Jacobian | $A_t^l=\partial u^l[t]/\partial u^l[t-1]$ | 单层转移的 Jacobian | 每层独立，形状 $(n_l, n_l)$ |
| 空间 Jacobian | $B_t^l=W^{l\leftarrow l-1}\,D_{\Psi,t}^{l-1}$ | `W[l-1] * Ψ'(layer_u[l-1][t]-V_th).unsqueeze(0)` | 形状 $(n_l, n_{l-1})$ |

---

## 三、使用规则

| 场景 | 用哪套记法 | 说明 |
|---|---|---|
| 论文、tex、研究报告、细节问题清单 | **论文记法** | $W^{l\leftarrow l-1}$、$u^l[t]$、$\delta^l[t]$ |
| 代码（`.py` 文件、docstring 正文、变量名） | **代码记法** | `weights[l]`、`layer_u[l]`、`delta[l]`、`dW[l]` |
| 本文档（对照备忘录） | **两套同时出现** | 唯一允许并排对照的地方 |

> **约定**：代码 docstring 开头可放一小段「论文↔代码」对照（指向本文档），但正文推导只用代码记法，不在两套之间反复切换。

---

## 四、待补充

> 后续挂新算法（OTTT / NDOT / e-prop / S-TLLR / TESS）时，把该算法引入的新符号（如资格迹 $\hat a$、learning signal、trace 向量等）的论文↔代码对应关系继续补进本文档第二节。

- [x] 阶段 1（BPTT-SG）完成：手写反向采用 detached-reset（$A_t=\lambda I$）与 full（$A_t=\lambda I-\lambda V_{th}D_{\Psi}$）两种模式，Gate A/C 各验两遍，相对误差 < 1e-5，全部 PASS（`python -m sanity.gradient_check`）。

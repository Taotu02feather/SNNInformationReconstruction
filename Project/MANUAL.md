# Project MANUAL —— SNN 本地学习逐层信息保留项目手册

> 本文档是 Project 目录的详细说明书，供第一次接触本项目的人快速理解现状与代码结构。
> 研究目标、阶段定义见 `GroupMeeting&CourseSlide/FutureDirections/FutureDirections_2026_10/Research_Goal_2026Oct.tex`。

---

## 一、项目现状盘点

### 1. 已实现内容（对应 Research_Goal 阶段）

- **LIF 神经元前向**（`snn/lif.py`）：阶段 0 网络骨架的一部分。前向（`reset` / `step`）已完成；反向（surrogate 接入 autograd）是阶段 1 的任务。
- **surrogate 梯度函数**（`snn/surrogate.py`）：阶段 0 网络骨架的一部分。函数已定义（fast-sigmoid 形式），尚未接入 autograd，接入是阶段 1 的任务。
- **多层 SNN 前向**（`network/mlp_snn.py`）：阶段 0 网络骨架的一部分。部分完成——只返回输出层脉冲，**缺**"返回每层 $u, s$"与"固定参数顺序"。
- **前向手算验证**（`sanity/forward_check.py`）：阶段 0 的验证手段。已跑通（n=2, T=3 手算对照 PASS + MLPSNN 形状检查 PASS）。
- **全局超参**（`utils/config.py`）：全局超参，不单独对应某个阶段。

### 2. 阶段 0 剩余任务

阶段 0 目标是"搭一套通用实验框架，架子上只挂一个能跑通的东西"。目前已完成 LIF / MLPSNN 前向这一部分，剩余：

1. **数据与脉冲编码模块**（DVS Gesture 为主、CIFAR10-DVS 为辅，$T=10$）—— **未开始**；
2. **统一网络骨架 SNN** 补全"返回每层 $u, s$"与"固定参数顺序"—— **部分完成**（`network/mlp_snn.py` 需扩展）；
3. **统一训练循环 `train_epoch()`**（支持 Accumulated / Online 两种模式）—— **未开始**；
4. **指标接口 `evaluate()`**（预留探针、CKA、梯度对齐的调用点）—— **未开始**。

### 3. 阶段 1（BPTT-SG）需要新建的文件

阶段 1 目标是"把 BPTT-SG 作为唯一算法插入骨架，跑出一条完整收敛曲线"。需要新建：

- **BPTT-SG 前向/反向**（建议新建 `algorithm/bptt_sg.py`）：手写 BPTT-SG 的 forward 与 backward，用形式 A 与 detached-reset。
- **Gate A/B/C 梯度验证**（建议新建 `sanity/gradient_check.py`）：验证手写梯度与 autograd 梯度一致（相对误差 $<10^{-5}$）。

**依赖的已有文件**：`snn/lif.py`、`snn/surrogate.py`、`network/mlp_snn.py`、`utils/config.py`。

### 4. 文件 → 阶段 → 状态 总表

| 文件 | 对应阶段 | 状态 |
|---|---|---|
| `snn/lif.py` | 阶段 0 网络骨架（前向） | ✅ 已完成（反向属阶段 1） |
| `snn/surrogate.py` | 阶段 0 网络骨架（函数） | ✅ 已完成（接入 autograd 属阶段 1） |
| `network/mlp_snn.py` | 阶段 0 网络骨架 | 🟡 部分完成（缺返回每层 u,s 与固定参数顺序） |
| `sanity/forward_check.py` | 阶段 0 验证手段 | ✅ 已完成（跑通 PASS） |
| `utils/config.py` | 全局超参 | ✅ 已完成（不单独对应阶段） |
| 数据/编码模块 | 阶段 0 | ❌ 未开始 |
| `train_epoch()` | 阶段 0 | ❌ 未开始 |
| `evaluate()` | 阶段 0 | ❌ 未开始 |
| `algorithm/bptt_sg.py` | 阶段 1 | ❌ 未开始 |
| `sanity/gradient_check.py` | 阶段 1 | ❌ 未开始 |

---

## 二、目录结构

```
Project/
├── MANUAL.md                # 本文件：项目手册（现状盘点 + 逐文件/逐函数说明）
├── README.md                # 项目说明、复现计划、待办清单
├── requirements.txt         # 依赖列表（torch、numpy）
├── snn/
│   ├── __init__.py          # 导出 LIF、fast_sigmoid_surrogate
│   ├── lif.py               # LIF 神经元（仅前向）
│   └── surrogate.py         # surrogate 梯度函数（fast-sigmoid）
├── network/
│   ├── __init__.py          # 导出 MLPSNN
│   └── mlp_snn.py           # 最简单的多层 SNN（仅前向）
├── sanity/
│   ├── __init__.py
│   └── forward_check.py     # 手算验证前向与代码一致
└── utils/
    ├── __init__.py
    └── config.py            # 全局超参（lambda、V_th 等）
```

---

## 三、逐文件说明

### `snn/lif.py`

- **文件路径**：`Project/snn/lif.py`
- **做什么**：实现 LIF（Leaky Integrate-and-Fire）神经元，只做前向、不做反向。维护膜电位 $u$ 与脉冲 $s$ 两个状态，按离散更新式逐步演化。
- **包含的类**：`LIF`（方法 `__init__`、`reset`、`step`）。
- **依赖**：`torch`、`torch.nn`；`utils.config`（导入 `LAMBDA`、`V_TH`）。
- **对应阶段**：阶段 0 网络骨架的一部分（前向已完成，反向属阶段 1）。

### `snn/surrogate.py`

- **文件路径**：`Project/snn/surrogate.py`
- **做什么**：定义 surrogate 梯度函数，用于后续反向传播替代不可导的脉冲发放导数。当前前向暂不使用。
- **包含的函数**：`fast_sigmoid_surrogate`。
- **依赖**：`torch`。
- **对应阶段**：阶段 0 网络骨架的一部分（函数已定义，接入 autograd 属阶段 1）。

### `network/mlp_snn.py`

- **文件路径**：`Project/network/mlp_snn.py`
- **做什么**：实现最简单的多层 LIF 前馈 SNN，输入 → 若干 LIF 层 → 输出，只做前向、不做学习。
- **包含的类**：`MLPSNN`（方法 `__init__`、`forward`）。
- **依赖**：`torch`、`torch.nn`；`snn.lif`（`LIF`）、`utils.config`（`LAMBDA`、`V_TH`）。
- **对应阶段**：阶段 0 网络骨架的一部分（部分完成，缺返回每层 $u,s$ 与固定参数顺序）。

### `sanity/forward_check.py`

- **文件路径**：`Project/sanity/forward_check.py`
- **做什么**：手算一个 n=2、T=3 的 LIF 例子，与代码前向结果对照；顺带检查 MLPSNN 输出形状。
- **包含的函数**：`main`。
- **依赖**：`torch`；`network.mlp_snn`（`MLPSNN`）、`snn.lif`（`LIF`）、`utils.config`（`LAMBDA`、`V_TH`）。
- **对应阶段**：阶段 0 的验证手段。
- **运行方式**：`python -m sanity.forward_check`。

### `utils/config.py`

- **文件路径**：`Project/utils/config.py`
- **做什么**：集中存放全局超参。
- **包含的常量**：`LAMBDA`、`V_TH`、`T`。
- **依赖**：无。
- **对应阶段**：全局超参，不单独对应某个阶段。

### 各 `__init__.py`

- `snn/__init__.py`：导出 `LIF`、`fast_sigmoid_surrogate`。
- `network/__init__.py`：导出 `MLPSNN`。
- `sanity/__init__.py`、`utils/__init__.py`：空包标记（仅模块说明注释）。

### `requirements.txt`

- 依赖列表：`torch`、`numpy`。运行环境建议用 `SNN_cuda`（torch 2.8.0+cu128，Python 3.10），因为默认 `.venv` 是 Python 3.14、torch 尚不支持。

---

## 四、逐函数说明

### 1. `LIF.__init__`

```python
def __init__(self, lambd: float = LAMBDA, v_th: float = V_TH)
```

- **参数**：
  - `lambd`（float）：膜电位衰减因子 $\lambda$，取值 $(0,1)$。默认取 `utils.config.LAMBDA = 0.5`。
  - `v_th`（float）：发放阈值 $V_{th}$。默认取 `utils.config.V_TH = 1.0`。
- **返回值**：无（构造实例）。
- **做什么**：初始化 LIF 神经元层，保存 $\lambda$、$V_{th}$，并声明状态 `self.u`（膜电位）、`self.s`（脉冲），初值为 `None`（由 `reset` 赋初值）。
- **对应公式**：无（仅保存超参）。
- **使用示例**：`lif = LIF(0.5, 1.0)`

### 2. `LIF.reset`

```python
def reset(self, shape)
```

- **参数**：
  - `shape`（tuple）：神经元状态形状，例如 `(n_neurons,)` 或 `(batch_size, n_neurons)`。
- **返回值**：无（原地设置 `self.u`、`self.s`）。
- **做什么**：把膜电位与脉冲初始化为零，即 $u[0]=0$、$s[0]=0$。
- **对应公式**：$u[0]=0,\quad s[0]=0$
- **使用示例**：`lif.reset((2,))`

### 3. `LIF.step`

```python
def step(self, current)
```

- **参数**：
  - `current`（torch.Tensor）：当前时刻输入电流 $I[t+1]$，形状与 `shape` 一致（如 `(batch, n_neurons)`）。
- **返回值**：`torch.Tensor`，当前时刻脉冲 $s[t+1]$，形状与 `current` 一致，元素为 0/1。
- **做什么**：单步前向。先更新膜电位，再据此产生脉冲。
- **对应公式**（形式 A，见细节问题清单 1.1）：
  $$u[t+1] = \lambda\big(u[t] - V_{th}s[t]\big) + I[t+1],\qquad s[t+1] = H(u[t+1] - V_{th})$$
- **使用示例**：`s = lif.step(current)  # current: (batch, n)`

### 4. `fast_sigmoid_surrogate`

```python
def fast_sigmoid_surrogate(u, v_th, beta=4.0)
```

- **参数**：
  - `u`（torch.Tensor）：膜电位，任意形状。
  - `v_th`（float）：阈值 $V_{th}$。
  - `beta`（float）：锐度参数 $\beta$，默认 4.0。
- **返回值**：`torch.Tensor`，surrogate 导数值，形状与 `u` 一致。
- **做什么**：计算 fast-sigmoid 形式的 surrogate 梯度，替代不可导的 $\partial s/\partial u$。
- **对应公式**：
  $$\Psi'(u - V_{th}) = \frac{1}{\big(1 + \beta|u - V_{th}|\big)^2}$$
- **使用示例**：`g = fast_sigmoid_surrogate(u, V_TH)`
- **TODO**：当前仅定义函数，未接入 autograd（接入是阶段 1 任务）。

### 5. `MLPSNN.__init__`

```python
def __init__(self, layer_sizes, lambd: float = LAMBDA, v_th: float = V_TH)
```

- **参数**：
  - `layer_sizes`（list[int]）：各层神经元数，如 `[n_input, n_hidden, n_output]`。
  - `lambd`（float）：$\lambda$，默认 `config.LAMBDA`。
  - `v_th`（float）：$V_{th}$，默认 `config.V_TH`。
- **返回值**：无（构造实例）。
- **做什么**：按 `layer_sizes` 构造多层 LIF 前馈 SNN。除输入层外每层一个 `LIF`；相邻层之间建一组全连接权重 `self.weights[l]`（形状 $(n_{out}, n_{in})$，随机初始化 $\times 0.1$）与偏置 `self.biases[l]`（形状 $(n_{out},)$）。
- **对应公式**：层间电流 $I^{l+1}[t] = W^{l+1\leftarrow l}\, s^l[t] + b^{l+1}$
- **使用示例**：`model = MLPSNN([2, 4, 2])`

### 6. `MLPSNN.forward`

```python
def forward(self, x)
```

- **参数**：
  - `x`（torch.Tensor）：外部输入序列，形状 `(T, batch_size, n_input)`。
- **返回值**：`torch.Tensor`，输出层脉冲序列，形状 `(T, batch_size, n_output)`。
- **做什么**：逐时间步做多层前向。每个时间步，输入层活动即外部输入 `x[t]`，然后逐层计算 `current = act @ weights[l].T + biases[l]` 并喂给该层 `LIF.step`。
- **对应公式**：每层 $s^l[t+1] = H\big(\lambda(u^l[t]-V_{th}s^l[t]) + W^{l\leftarrow l-1}s^{l-1}[t+1] + b^l - V_{th}\big)$ 的离散递推（详见 `LIF.step`）。
- **使用示例**：`out = model(x)  # x: (T, B, n_input)`
- **TODO（阶段 0）**：当前只返回输出层脉冲，**缺**"返回每层 $u, s$"（供探针/CKA 使用）与"固定参数顺序"（供梯度对齐展平使用）。

### 7. `main`（`sanity/forward_check.py`）

```python
def main()
```

- **参数**：无。
- **返回值**：无（打印结果；不一致时 `raise SystemExit(1)`）。
- **做什么**：手算 n=2、T=3 的 LIF 前向期望值，与代码前向对照；再检查 MLPSNN 输出形状。
- **对应公式**：`LIF.step` 的离散更新式。
- **使用示例**：`python -m sanity.forward_check`

---

## 五、未实现文件清单

以下文件当前**未实现但计划实现**：

| 文件路径（计划） | 对应阶段 | 依赖的已有文件 |
|---|---|---|
| 数据与脉冲编码模块（如 `data/` 或 `utils/dataset.py`） | 阶段 0 | 无（或 `torch`） |
| 统一训练循环（如 `utils/train.py` 的 `train_epoch()`） | 阶段 0 | `network/mlp_snn.py`、`snn/lif.py` |
| 指标接口（如 `utils/evaluate.py` 的 `evaluate()`） | 阶段 0 | `network/mlp_snn.py` |
| `algorithm/bptt_sg.py`（BPTT-SG 前向/反向） | 阶段 1 | `snn/lif.py`、`snn/surrogate.py`、`network/mlp_snn.py` |
| `sanity/gradient_check.py`（Gate A/B/C） | 阶段 1 | `algorithm/bptt_sg.py`、`network/mlp_snn.py` |

---

## 六、数学符号与代码对应表

| 数学符号 | 含义 | 代码变量名 | 出现位置 |
|---|---|---|---|
| $\lambda$ | 膜衰减因子 | `lambd`（实例属性）、`LAMBDA`（常量） | `snn/lif.py`、`utils/config.py` |
| $V_{th}$ | 发放阈值 | `v_th`（实例属性）、`V_TH`（常量） | `snn/lif.py`、`utils/config.py` |
| $u$（$u_t$） | 膜电位 | `u`（`LIF.u`） | `snn/lif.py` |
| $s$（$s_t$） | 脉冲 | `s`（`LIF.s`） | `snn/lif.py` |
| $I[t]$ | 输入电流 | `current`（`LIF.step` 参数） | `snn/lif.py` |
| $\beta$ | surrogate 锐度参数 | `beta`（`fast_sigmoid_surrogate` 参数） | `snn/surrogate.py` |
| $\Psi'(\cdot)$ | surrogate 梯度 | `fast_sigmoid_surrogate` | `snn/surrogate.py` |
| $W^{l\leftarrow l-1}$ | 层间权重 | `weights[l]` | `network/mlp_snn.py` |
| $b^l$ | 层间偏置 | `biases[l]` | `network/mlp_snn.py` |
| $T$ | 时间步数 | `T` | `utils/config.py`、`sanity/forward_check.py` |
| $n$ | 神经元数 | `n` | `sanity/forward_check.py` |



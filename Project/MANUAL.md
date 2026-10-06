# Project MANUAL —— SNN 本地学习逐层信息保留项目手册

> 本文档是 Project 目录的详细说明书，供第一次接触本项目的人快速理解现状与代码结构。
> 研究目标、阶段定义见 `GroupMeeting&CourseSlide/FutureDirections/FutureDirections_2026_10/Research_Goal_2026Oct.tex`。

---

## 一、项目现状盘点

### 1. 已实现内容（对应 Research_Goal 阶段）

- **LIF 神经元前向 + 反向**（`snn/lif.py`）：阶段 0 网络骨架。前向（`reset` / `step`）与反向（surrogate 接入 autograd）均已完成。
- **surrogate 梯度函数 + SpikeFunction**（`snn/surrogate.py`）：阶段 0 网络骨架。`fast_sigmoid_surrogate`（纯函数）与 `SpikeFunction`（自定义 autograd 算子：前向 Heaviside、反向 surrogate）均已实现。
- **多层 SNN**（`network/mlp_snn.py`）：阶段 0 网络骨架。`forward`（前向）、`forward_states`（返回每层 u,s）、`param_vector`（按层交替展平）均已完成。
- **数据与脉冲编码**（`data/encoding.py`、`data/dataset.py`）：阶段 0 任务 1。泊松编码 `poisson_encode` 已实现；真实 DVS 数据集加载留 TODO（用可分合成数据 `synthetic_dataset` 演示）。
- **训练循环**（`utils/train.py`）：阶段 0 任务 3。`train_epoch` 已实现（firing rate 损失 + surrogate 反向）。
- **指标接口**（`utils/evaluate.py`）：阶段 0 任务 4。`evaluate` 已实现（探针/CKA/梯度对齐调用点留阶段 2）。
- **验证**（`sanity/forward_check.py`、`sanity/stage0_check.py`）：前向手算 + surrogate 反向贯通 + 阶段 0 全链路，均 PASS。
- **全局超参**（`utils/config.py`）：全局超参，不单独对应某个阶段。

### 2. 阶段 0 剩余任务

阶段 0 目标"搭一套通用实验框架，架子上只挂一个能跑到头（前向 + 反向贯通）的东西"已基本达成（数据编码、网络骨架、训练循环、指标接口、surrogate 接 autograd 均已实现，并通过 `sanity/stage0_check.py` 验证）。剩余：

1. **真实 DVS Gesture / CIFAR10-DVS 数据加载**（`data/dataset.py` 的 `load_dvs_gesture` 目前抛 `NotImplementedError`）—— **未开始**（阶段 0 用可分合成数据演示，真实数据在阶段 5 评估实验前接入即可）。

### 3. 阶段 1（BPTT-SG）需要新建的文件

阶段 1 目标是"把 BPTT-SG 作为唯一算法插入骨架，跑出一条完整收敛曲线"。需要新建：

- **BPTT-SG 前向/反向**（建议新建 `algorithm/bptt_sg.py`）：手写 BPTT-SG 的 forward 与 backward，用形式 A 与 detached-reset。
- **Gate A/B/C 梯度验证**（建议新建 `sanity/gradient_check.py`）：验证手写梯度与 autograd 梯度一致（相对误差 $<10^{-5}$）。

**依赖的已有文件**：`snn/lif.py`、`snn/surrogate.py`、`network/mlp_snn.py`、`utils/config.py`。

### 4. 文件 → 阶段 → 状态 总表

| 文件 | 对应阶段 | 状态 |
|---|---|---|
| `snn/lif.py` | 阶段 0 网络骨架 | ✅ 已完成（前向 + 反向） |
| `snn/surrogate.py` | 阶段 0 网络骨架 | ✅ 已完成（SpikeFunction 接入） |
| `network/mlp_snn.py` | 阶段 0 网络骨架 | ✅ 已完成（forward + forward_states + param_vector） |
| `data/encoding.py` | 阶段 0 数据编码 | ✅ 已完成（泊松编码） |
| `data/dataset.py` | 阶段 0 数据集 | 🟡 部分完成（合成数据可用，DVS 加载 TODO） |
| `utils/train.py` | 阶段 0 训练循环 | ✅ 已完成（train_epoch） |
| `utils/evaluate.py` | 阶段 0 指标接口 | ✅ 已完成（evaluate） |
| `sanity/forward_check.py` | 阶段 0 验证 | ✅ 已完成 |
| `sanity/stage0_check.py` | 阶段 0 验证 | ✅ 已完成 |
| `utils/config.py` | 全局超参 | ✅ 已完成 |
| `algorithm/bptt_sg.py` | 阶段 1 | ❌ 未开始 |
| `sanity/gradient_check.py` | 阶段 1 | ❌ 未开始 |

---

## 二、目录结构

```
Project/
├── MANUAL.md                # 本文件：项目手册（现状盘点 + 逐文件/逐函数说明）
├── README.md                # 项目说明、复现计划、待办清单
├── requirements.txt         # 依赖列表（torch、numpy）
├── data/
│   ├── __init__.py
│   ├── encoding.py          # 脉冲编码（泊松编码）
│   └── dataset.py           # 数据集接口（合成数据 + DVS TODO）
├── snn/
│   ├── __init__.py          # 导出 LIF、fast_sigmoid_surrogate、SpikeFunction
│   ├── lif.py               # LIF 神经元（前向 Heaviside + 反向 surrogate）
│   └── surrogate.py         # surrogate 梯度 + SpikeFunction 算子
├── network/
│   ├── __init__.py          # 导出 MLPSNN
│   └── mlp_snn.py           # 多层 SNN（forward + forward_states + param_vector）
├── sanity/
│   ├── __init__.py
│   ├── forward_check.py     # 手算验证前向 + surrogate 反向贯通
│   └── stage0_check.py      # 阶段 0 框架全链路验证
└── utils/
    ├── __init__.py
    ├── config.py            # 全局超参（lambda、V_th、beta 等）
    ├── train.py             # 训练循环 train_epoch
    └── evaluate.py          # 指标接口 evaluate
```

---

## 三、逐文件说明

### `snn/lif.py`

- **文件路径**：`Project/snn/lif.py`
- **做什么**：实现 LIF（Leaky Integrate-and-Fire）神经元，前向 Heaviside、反向走 surrogate。维护膜电位 $u$ 与脉冲 $s$ 两个状态，按离散更新式逐步演化。
- **包含的类**：`LIF`（方法 `__init__`、`reset`、`step`）。
- **依赖**：`torch`、`torch.nn`；`snn.surrogate`（`SpikeFunction`）、`utils.config`（`LAMBDA`、`V_TH`、`BETA`）。
- **对应阶段**：阶段 0 网络骨架（前向 + 反向均已完成）。

### `snn/surrogate.py`

- **文件路径**：`Project/snn/surrogate.py`
- **做什么**：定义 surrogate 梯度函数，以及自定义 autograd 算子 `SpikeFunction`（前向 Heaviside、反向 surrogate），让脉冲发放可导。
- **包含的函数/类**：`fast_sigmoid_surrogate`（纯函数）、`SpikeFunction`（`torch.autograd.Function`）。
- **依赖**：`torch`。
- **对应阶段**：阶段 0 网络骨架（已实现）。

### `network/mlp_snn.py`

- **文件路径**：`Project/network/mlp_snn.py`
- **做什么**：实现多层 LIF 前馈 SNN。`forward` 做前向，`forward_states` 额外返回每层 $u,s$，`param_vector` 按层交替顺序展平参数。
- **包含的类**：`MLPSNN`（方法 `__init__`、`forward`、`forward_states`、`param_vector`）。
- **依赖**：`torch`、`torch.nn`；`snn.lif`（`LIF`）、`utils.config`（`LAMBDA`、`V_TH`）。
- **对应阶段**：阶段 0 网络骨架（已完成）。

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
- **包含的常量**：`LAMBDA`、`V_TH`、`BETA`、`T`。
- **依赖**：无。
- **对应阶段**：全局超参，不单独对应某个阶段。

### `data/encoding.py`

- **文件路径**：`Project/data/encoding.py`
- **做什么**：脉冲编码。`poisson_encode` 把 [0,1] 连续输入按泊松采样编码成 T 步脉冲序列。
- **包含的函数**：`poisson_encode`。
- **依赖**：`torch`。
- **对应阶段**：阶段 0 任务 1（数据与脉冲编码）。

### `data/dataset.py`

- **文件路径**：`Project/data/dataset.py`
- **做什么**：数据集接口。`synthetic_dataset` 生成可分合成数据（每类一个模板 + 噪声）用于演示；`load_dvs_gesture` 是真实数据加载的 TODO。
- **包含的函数**：`synthetic_dataset`、`load_dvs_gesture`。
- **依赖**：`torch`。
- **对应阶段**：阶段 0 任务 1（真实 DVS 加载留 TODO）。

### `utils/train.py`

- **文件路径**：`Project/utils/train.py`
- **做什么**：训练循环 `train_epoch`（脉冲编码 → 前向 → firing rate 损失 → surrogate 反向 → 更新）。
- **包含的函数**：`train_epoch`。
- **依赖**：`torch`、`torch.nn.functional`；`data.encoding`（`poisson_encode`）。
- **对应阶段**：阶段 0 任务 3（训练循环）。

### `utils/evaluate.py`

- **文件路径**：`Project/utils/evaluate.py`
- **做什么**：指标接口 `evaluate`（分类准确率，探针/CKA/梯度对齐调用点留阶段 2）。
- **包含的函数**：`evaluate`。
- **依赖**：`torch`；`data.encoding`（`poisson_encode`）。
- **对应阶段**：阶段 0 任务 4（指标接口）。

### `sanity/stage0_check.py`

- **文件路径**：`Project/sanity/stage0_check.py`
- **做什么**：阶段 0 框架全链路验证（合成数据 → 编码 → 训练 → 评估 → forward_states → param_vector）。
- **包含的函数**：`main`。
- **依赖**：`torch`；`data.dataset`、`data.encoding`、`network.mlp_snn`、`utils.train`、`utils.evaluate`。
- **对应阶段**：阶段 0 验证手段。
- **运行方式**：`python -m sanity.stage0_check`。

### 各 `__init__.py`

- `snn/__init__.py`：导出 `LIF`、`fast_sigmoid_surrogate`、`SpikeFunction`。
- `network/__init__.py`：导出 `MLPSNN`。
- `data/__init__.py`、`sanity/__init__.py`、`utils/__init__.py`：空包标记（仅模块说明注释）。

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
- **状态**：已实现。配套的 `SpikeFunction`（`torch.autograd.Function`，前向 Heaviside、反向 $\Psi'$）已接入 `LIF.step`，让脉冲发放可导。

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
- **说明**：`forward` 只返回输出层脉冲；需要每层 $u,s$ 用 `forward_states`，需要按层交替展平参数用 `param_vector`（均已实现，见下）。

### 7. `main`（`sanity/forward_check.py`）

```python
def main()
```

- **参数**：无。
- **返回值**：无（打印结果；不一致时 `raise SystemExit(1)`）。
- **做什么**：手算 n=2、T=3 的 LIF 前向期望值，与代码前向对照；再检查 MLPSNN 输出形状。
- **对应公式**：`LIF.step` 的离散更新式。
- **使用示例**：`python -m sanity.forward_check`

### 8. 新增函数简要说明（阶段 0 补充）

- **`SpikeFunction`（`snn/surrogate.py`）**：`torch.autograd.Function`，前向 `s = (u >= v_th).float()`，反向 `dL/du = (dL/ds)·Ψ'(u-v_th)`。对应 $\Psi'(x)=\frac{1}{(1+\beta|x|)^2}$。示例：`s = SpikeFunction.apply(u, v_th, beta)`。
- **`MLPSNN.forward_states(x)`**：前向并返回 `(out, layer_u, layer_s)`，含每层膜电位与脉冲序列。供阶段 2 探针/CKA 使用。
- **`MLPSNN.param_vector()`**：按层交替顺序 $[W_0, b_0, W_1, b_1, \dots]$ 展平所有参数为 1D 向量（行优先）。供阶段 2 梯度对齐"按层切片"使用。
- **`poisson_encode(x, T)`（`data/encoding.py`）**：泊松编码，`(batch,...)` → `(T, batch,...)` 脉冲。对应 $s \sim \mathrm{Bernoulli}(x)$。
- **`synthetic_dataset(n_samples, n_input, n_classes, seed)`（`data/dataset.py`）**：可分合成数据（每类模板 + 噪声）。供阶段 0 演示。
- **`train_epoch(model, x, y, optimizer, T, mode)`（`utils/train.py`）**：单 epoch 训练（编码 → 前向 → firing rate 损失 → surrogate 反向 → 更新）。
- **`evaluate(model, x, y, T)`（`utils/evaluate.py`）**：分类准确率，预留探针/CKA/梯度对齐调用点。

---

## 五、未实现文件清单

以下文件当前**未实现但计划实现**：

| 文件路径（计划） | 对应阶段 | 依赖的已有文件 |
|---|---|---|
| DVS Gesture / CIFAR10-DVS 数据加载（`data/dataset.py` 的 `load_dvs_gesture`） | 阶段 0（可选，阶段 5 前接入） | `torch`（或 tonic 等数据集库） |
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
| $\beta$ | surrogate 锐度参数 | `beta`（参数）、`BETA`（常量） | `snn/surrogate.py`、`snn/lif.py`、`utils/config.py` |
| $\Psi'(\cdot)$ | surrogate 梯度 | `fast_sigmoid_surrogate` | `snn/surrogate.py` |
| $W^{l\leftarrow l-1}$ | 层间权重 | `weights[l]` | `network/mlp_snn.py` |
| $b^l$ | 层间偏置 | `biases[l]` | `network/mlp_snn.py` |
| $T$ | 时间步数 | `T` | `utils/config.py`、`sanity/forward_check.py` |
| $n$ | 神经元数 | `n` | `sanity/forward_check.py` |



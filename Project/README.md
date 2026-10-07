# Project：SNN 本地学习逐层信息保留 —— 基础代码

本目录是"SNN 本地学习的逐层信息保留与损失定位"研究的基础代码库，当前已完成 **阶段 0（通用实验框架）+ 阶段 1（BPTT-SG）**。逐文件/逐函数的详细说明见 **`MANUAL.md`**（本文件只做概览与复现计划）。

## 项目结构

```
Project/
├── MANUAL.md                # 项目手册（现状盘点 + 逐文件/逐函数说明）
├── README.md                # 本文件：项目说明、复现计划、待办清单
├── SYMBOL_MAPPING.md        # 论文记法 ↔ 代码记法 对照备忘录（差 1 层的映射）
├── requirements.txt         # 依赖列表（torch、numpy）
├── data/
│   ├── __init__.py
│   ├── encoding.py          # 脉冲编码（泊松编码）
│   └── dataset.py           # 数据集接口（合成数据 + DVS TODO）
├── snn/
│   ├── __init__.py          # 导出 LIF、fast_sigmoid_surrogate、SpikeFunction
│   ├── lif.py               # LIF 神经元（前向 Heaviside + 反向 surrogate；step 支持 detach_reset）
│   └── surrogate.py         # surrogate 梯度 + SpikeFunction 算子
├── network/
│   ├── __init__.py          # 导出 MLPSNN
│   └── mlp_snn.py           # 多层 SNN（forward + forward_states + param_vector；forward 支持 detach_reset）
├── algorithm/
│   ├── __init__.py          # 导出 bptt_sg_forward / bptt_sg_backward / bptt_sg_step
│   └── bptt_sg.py           # 手写 BPTT-SG 前向/反向（阶段 1）
├── sanity/
│   ├── __init__.py
│   ├── forward_check.py     # 手算验证前向与代码一致
│   ├── stage0_check.py      # 阶段 0 全链路验证
│   └── gradient_check.py    # 阶段 1 Gate A/B/C 梯度验证 + 收敛曲线
└── utils/
    ├── __init__.py
    ├── config.py            # 全局超参（lambda、V_th 等）
    ├── train.py             # 训练循环（train_epoch + train_epoch_bptt_sg）
    └── evaluate.py          # 指标接口（evaluate，探针/CKA/梯度对齐调用点留阶段 2）
```

## 已完成内容

### 阶段 0：通用实验框架

- LIF 神经元（`snn/lif.py`）：前向（`reset` / `step`）+ 反向（surrogate 接入 autograd）。
- surrogate 梯度（`snn/surrogate.py`）：`fast_sigmoid_surrogate` 纯函数 + `SpikeFunction`（前向 Heaviside、反向 $\Psi'$）。
- 多层 SNN（`network/mlp_snn.py`）：`forward`（前向）、`forward_states`（返回每层 u,s）、`param_vector`（按层交替展平）。
- 数据与编码（`data/`）：泊松编码 `poisson_encode` + 可分合成数据 `synthetic_dataset`（真实 DVS 加载留 TODO）。
- 训练循环（`utils/train.py`）：`train_epoch`（firing rate 损失 + surrogate 反向）。
- 指标接口（`utils/evaluate.py`）：`evaluate`（探针/CKA/梯度对齐调用点留阶段 2）。
- 验证（`sanity/forward_check.py`、`sanity/stage0_check.py`）：前向手算 + 反向贯通 + 阶段 0 全链路，均 PASS。

### 阶段 1：BPTT-SG

- 手写 BPTT-SG（`algorithm/bptt_sg.py`）：形式 A，支持 detached-reset（默认）与 full 两种模式（`detach_reset` 开关）。
- Gate A/B/C 梯度验证（`sanity/gradient_check.py`）：手写 vs autograd 相对误差 < 1e-5（实测 1e-7~1e-8）。
- 收敛曲线：full BPTT 在 256 样本合成数据上 last3 均值 > 0.85；detached-reset 作对照。
- 接口扩展：`LIF.step` / `MLPSNN.forward` / `forward_states` 加 `detach_reset`（默认 False，向后兼容）。

## 运行方式

在 `Project/` 目录下：

```bash
pip install -r requirements.txt
python -m sanity.forward_check     # 前向手算验证
python -m sanity.stage0_check      # 阶段 0 全链路
python -m sanity.gradient_check    # 阶段 1 Gate A/B/C + 收敛曲线
```

## 下一步（按阶段）

- **阶段 2**：焊上探针 / CKA / 梯度对齐的钩子，并在 BPTT 自身上验证工具本身（$\mathrm{CKA}=1$，$\cos=1$）。
- **阶段 3**：挂载 OTTT / NDOT / e-prop（自己实现）。
- **阶段 4**：接入 S-TLLR / TESS（外部验证，优先用作者公开代码）。
- **阶段 5**：逐层评估实验（探针准确率 / CKA / 梯度对齐三张图 + 归因 + Pareto）。
- **阶段 6**：与硬件方向对接（状态 / 通信 / 更新清单）。

完整阶段定义与任务拆解见 `Research_Goal_2026Oct.tex`。

## 复现计划（文字说明）

要复现的算法：BPTT-SG、OTTT、NDOT、e-prop、S-TLLR、TESS，分三个层次：

- **自己实现**：BPTT-SG、OTTT、NDOT。
- **必须验证**：e-prop。
- **外部验证**：S-TLLR、TESS（优先使用作者公开代码）。

每个算法的复现目标是"保证近似对象与基准对象在数学上一致"，例如：OTTT 的 trace 递推与闭式求和一致、NDOT 的 trace 递推与历史展开式一致、e-prop 的 eligibility-trace / learning-signal 分解恒等成立，而不是单纯比较某个准确率数字。

## 评估方法（文字说明，摘自研究计划）

本地学习（OTTT、NDOT、S-TLLR、TESS）每层优化局部目标、没有统一全局梯度，现有评估只看最终准确率，无法定位信息在哪一层损失。为此在层与层之间引入中间量，三个研究方向：

- **逐层探针**：在每层挂独立解码器，看该层表示能解码多少任务信息（信息是否还在）。
- **逐层表示相似度**：比较本地学习与 BPTT 对应层表示的几何结构（表示是否被扭曲）。
- **逐层梯度对齐**：比较本地学习与 BPTT 对应层梯度方向（优化方向是否偏离）。

三者构成"梯度（过程）→ 表示（结果）→ 任务信息（效用）"的因果链。

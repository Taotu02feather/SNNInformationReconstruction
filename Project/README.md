# Project：SNN 本地学习逐层信息保留 —— 基础代码

本目录是"SNN 本地学习的逐层信息保留与损失定位"研究的基础代码库，当前处于**最小可运行基础**阶段：只搭好 LIF 神经元与多层 SNN 的**前向**，尚未实现任何训练算法。

## 项目结构

```
Project/
├── README.md                # 本文件：项目说明、复现计划、待办清单
├── requirements.txt         # 依赖列表
├── snn/
│   ├── __init__.py
│   ├── lif.py               # LIF 神经元，只做前向
│   └── surrogate.py         # surrogate 梯度函数（为反向预留，前向暂不用）
├── network/
│   ├── __init__.py
│   └── mlp_snn.py           # 最简单的多层 SNN（只有前向）
├── sanity/
│   ├── __init__.py
│   └── forward_check.py     # 手算验证前向与代码一致
└── utils/
    ├── __init__.py
    └── config.py            # 全局超参（lambda、V_th 等）
```

## 本次完成了什么

- LIF 神经元（`snn/lif.py`）：只做前向，维护膜电位 $u[t]$ 与脉冲 $s[t]$，不做反向。
- surrogate 梯度函数（`snn/surrogate.py`）：fast-sigmoid 形式，为后续反向传播预留。
- 多层 SNN（`network/mlp_snn.py`）：输入 → 若干 LIF 层 → 输出，只做前向。
- 前向验证（`sanity/forward_check.py`）：手算一个 $n=2, T=3$ 的 LIF 例子与代码对照。
- 全局超参（`utils/config.py`）：集中管理 $\lambda$、$V_{th}$ 等。

## 运行方式

在 `Project/` 目录下：

```bash
pip install -r requirements.txt
python -m sanity.forward_check
```

## 下一步要加什么（按依赖顺序）

1. **surrogate 反向 + autograd 接入**：把 `surrogate.py` 接入 `LIF` 的 backward，使脉冲发放可导。
2. **BPTT-SG**（自己实现）：时间展开 + surrogate 反传，作为 reference estimator $g_{\mathrm{BPTT}}$。
3. **OTTT**（自己实现）：固定衰减 trace $\hat a^l[t]=\lambda\hat a^l[t-1]+s^l[t]$。
4. **NDOT**（自己实现）：动态系数 $e^l[t]$ 替代固定 $\lambda$。
5. **e-prop**（必须验证）：eligibility-trace / learning-signal 分解。
6. **逐层评估中间量**：逐层探针、逐层表示相似度、逐层梯度对齐（对应三个研究方向）。
7. **数据集加载与训练循环**。

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

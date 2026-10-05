"""最简单的多层 SNN（仅前向，不做学习）。

TODO（阶段 0）：当前 forward 只返回输出层脉冲，缺"返回每层 u, s"与"固定参数顺序"。
"""

import torch
import torch.nn as nn

from snn.lif import LIF
from utils.config import LAMBDA, V_TH


class MLPSNN(nn.Module):
    """多层 LIF 前馈 SNN。

    结构由 layer_sizes 决定，例如 [n_input, n_hidden, n_output]。
    输入层直接接收外部输入，其余每层是一个 LIF 神经元层。

    层间连接（线性）：
        I^{l+1}[t] = W^{l+1←l} s^l[t] + b^{l+1}
    每层内部按 LIF 离散更新（见 snn/lif.py）。

    属性：
        lif_layers: 除输入层外每层一个 LIF 实例（nn.ModuleList）
        weights: 层间权重，weights[l] 形状 (n_out, n_in)（nn.ParameterList）
        biases: 层间偏置，biases[l] 形状 (n_out,)（nn.ParameterList）
    """

    def __init__(self, layer_sizes, lambd: float = LAMBDA, v_th: float = V_TH):
        """按 layer_sizes 构造多层 LIF 前馈 SNN。

        参数:
            layer_sizes (list[int]): 各层神经元数，如 [n_input, n_hidden, n_output]。
            lambd (float): 膜衰减因子 λ，默认 config.LAMBDA=0.5。
            v_th (float): 发放阈值 V_th，默认 config.V_TH=1.0。

        返回:
            无（构造实例）。

        使用示例:
            model = MLPSNN([2, 4, 2])
        """
        super().__init__()
        self.layer_sizes = list(layer_sizes)
        # 除输入层（第 0 层）外，每一层各有一个 LIF 神经元层
        self.lif_layers = nn.ModuleList(
            [LIF(lambd, v_th) for _ in self.layer_sizes[1:]]
        )
        # 层间全连接权重与偏置：weights[l] 连接第 l 层 → 第 l+1 层
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()
        for n_in, n_out in zip(self.layer_sizes[:-1], self.layer_sizes[1:]):
            # 权重形状 (n_out, n_in)，随机初始化并缩小到 0.1 量级
            self.weights.append(nn.Parameter(torch.randn(n_out, n_in) * 0.1))
            self.biases.append(nn.Parameter(torch.zeros(n_out)))

    def forward(self, x):
        """逐时间步做多层前向。

        参数:
            x (torch.Tensor): 外部输入序列，形状 (T, batch_size, n_input)。

        返回:
            torch.Tensor: 输出层脉冲序列，形状 (T, batch_size, n_output)。

        使用示例:
            out = model(x)  # x: (T, B, n_input)

        TODO（阶段 0）: 当前只返回输出层脉冲，缺"返回每层 u, s"（供探针/CKA 使用）
            与"固定参数顺序"（供梯度对齐展平使用）。
        """
        T, batch, _ = x.shape
        n_layers = len(self.layer_sizes)

        # 初始化每层状态：u^l[0] = 0, s^l[0] = 0
        for l in range(n_layers - 1):
            self.lif_layers[l].reset((batch, self.layer_sizes[l + 1]))

        outputs = []
        for t in range(T):
            act = x[t]  # 输入层（第 0 层）活动即外部输入，形状 (batch, n_input)
            for l in range(n_layers - 1):
                # 线性变换得到第 l+1 层的输入电流 I^{l+1}[t] = W s^l[t] + b
                current = act @ self.weights[l].T + self.biases[l]
                # 该层 LIF 单步更新，输出其脉冲作为下一层的输入
                act = self.lif_layers[l].step(current)
            outputs.append(act)  # 记录当前时间步的输出层脉冲
        # 把 T 个时间步的输出堆叠成 (T, batch, n_output)
        return torch.stack(outputs, dim=0)

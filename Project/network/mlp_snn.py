"""最简单的多层 SNN（仅前向，不做学习）。"""

import torch
import torch.nn as nn

from snn.lif import LIF
from utils.config import LAMBDA, V_TH


class MLPSNN(nn.Module):
    """多层 LIF 前馈 SNN。

    结构由 layer_sizes 决定，例如 [n_input, n_hidden, n_output]。
    输入层直接接收外部输入，其余每层是一个 LIF 神经元层。
    """

    def __init__(self, layer_sizes, lambd: float = LAMBDA, v_th: float = V_TH):
        super().__init__()
        self.layer_sizes = list(layer_sizes)
        # 除输入层外的每一层各有一个 LIF
        self.lif_layers = nn.ModuleList(
            [LIF(lambd, v_th) for _ in self.layer_sizes[1:]]
        )
        # 层间权重与偏置
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()
        for n_in, n_out in zip(self.layer_sizes[:-1], self.layer_sizes[1:]):
            self.weights.append(nn.Parameter(torch.randn(n_out, n_in) * 0.1))
            self.biases.append(nn.Parameter(torch.zeros(n_out)))

    def forward(self, x):
        """前向传播。

        x: (T, batch_size, n_input) 的外部输入序列
        返回: (T, batch_size, n_output) 的输出脉冲序列
        """
        T, batch, _ = x.shape
        n_layers = len(self.layer_sizes)

        # 初始化每层状态 u^l[0] = 0, s^l[0] = 0
        for l in range(n_layers - 1):
            self.lif_layers[l].reset((batch, self.layer_sizes[l + 1]))

        outputs = []
        for t in range(T):
            act = x[t]  # 输入层活动即外部输入 (batch, n_input)
            for l in range(n_layers - 1):
                current = act @ self.weights[l].T + self.biases[l]
                act = self.lif_layers[l].step(current)
            outputs.append(act)
        return torch.stack(outputs, dim=0)

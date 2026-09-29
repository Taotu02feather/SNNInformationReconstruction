"""LIF（Leaky Integrate-and-Fire）神经元，仅实现前向。

离散更新（OTTT/NDOT 约定的形式）：
    u[t+1] = lambda * (u[t] - V_th * s[t]) + I[t+1]
    s[t+1] = H(u[t+1] - V_th)
其中 H 为 Heaviside 阶跃函数，I[t] 为输入电流。
"""

import torch
import torch.nn as nn

from utils.config import LAMBDA, V_TH


class LIF(nn.Module):
    """LIF 神经元层，维护膜电位与脉冲状态，只做前向、不做反向。"""

    def __init__(self, lambd: float = LAMBDA, v_th: float = V_TH):
        super().__init__()
        self.lambd = lambd
        self.v_th = v_th
        self.u = None  # 膜电位 u[t]
        self.s = None  # 脉冲 s[t]

    def reset(self, shape):
        """初始化状态 u[0] = 0, s[0] = 0。

        shape: 神经元形状，例如 (n_neurons,) 或 (batch_size, n_neurons)。
        """
        self.u = torch.zeros(shape)
        self.s = torch.zeros(shape)

    def step(self, current):
        """单步前向：输入当前时刻电流 I[t+1]，输出脉冲 s[t+1]。

        current: 与神经元形状一致的输入电流张量。
        """
        self.u = self.lambd * (self.u - self.v_th * self.s) + current
        self.s = (self.u >= self.v_th).float()
        return self.s

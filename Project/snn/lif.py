"""LIF（Leaky Integrate-and-Fire）神经元，仅实现前向。

离散更新（OTTT/NDOT 约定的"形式 A"，见细节问题清单 1.1）：
    u[t+1] = lambda * (u[t] - V_th * s[t]) + I[t+1]
    s[t+1] = H(u[t+1] - V_th)
其中 H 为 Heaviside 阶跃函数，I[t] 为输入电流。

反向：脉冲发放用 surrogate 梯度替代不可导的 H'（见 snn.surrogate.SpikeFunction），
因此本文件前向是 Heaviside、反向走 surrogate（阶段 0 任务）。
"""

import torch
import torch.nn as nn

from snn.surrogate import SpikeFunction
from utils.config import LAMBDA, V_TH, BETA


class LIF(nn.Module):
    """LIF 神经元层，维护膜电位与脉冲状态；前向 Heaviside，反向走 surrogate。

    状态：
        self.u: 膜电位 u[t]，由 reset() 初始化为 0
        self.s: 脉冲 s[t]，由 reset() 初始化为 0

    离散更新（形式 A）：
        u[t+1] = lambda * (u[t] - V_th * s[t]) + I[t+1]
        s[t+1] = H(u[t+1] - V_th)
    """

    def __init__(self, lambd: float = LAMBDA, v_th: float = V_TH, beta: float = BETA):
        """初始化 LIF 神经元层。

        参数:
            lambd (float): 膜电位衰减因子 lambda，取值 (0, 1)。默认 config.LAMBDA=0.5。
            v_th (float): 发放阈值 V_th。默认 config.V_TH=1.0。
            beta (float): surrogate 锐度参数 β。默认 config.BETA=4.0。

        返回:
            无（构造实例）。

        使用示例:
            lif = LIF(0.5, 1.0)
        """
        super().__init__()
        self.lambd = lambd  # 膜衰减因子 lambda
        self.v_th = v_th    # 发放阈值 V_th
        self.beta = beta    # surrogate 锐度参数 β
        self.u = None       # 膜电位 u[t]，待 reset() 初始化
        self.s = None       # 脉冲 s[t]，待 reset() 初始化

    def reset(self, shape):
        """把膜电位与脉冲初始化为零：u[0]=0, s[0]=0。

        参数:
            shape (tuple): 神经元形状，例如 (n_neurons,) 或 (batch_size, n_neurons)。

        返回:
            无（原地设置 self.u、self.s）。

        使用示例:
            lif.reset((2,))
        """
        self.u = torch.zeros(shape)  # u[0] = 0
        self.s = torch.zeros(shape)  # s[0] = 0

    def step(self, current, detach_reset=False):
        """单步前向：输入当前时刻电流 I[t+1]，输出脉冲 s[t+1]。

        参数:
            current (torch.Tensor): 当前时刻输入电流 I[t+1]，形状与 shape 一致
                （如 (batch, n_neurons)）。
            detach_reset (bool): 是否 detach 复位项（默认 False）。
                False = full BPTT，反向时间 Jacobian A_t = λI - λV_th·D_Ψ；
                True  = detached-reset，复位项 s.detach()，A_t = λI（丢复位路径）。
                阶段 1 手写 BPTT-SG 与 gradcheck 的 autograd 参考侧均需此开关。

        返回:
            torch.Tensor: 当前时刻脉冲 s[t+1]，形状同 current，元素为 0/1。

        对应公式（形式 A）:
            u[t+1] = lambda * (u[t] - V_th * s[t]) + I[t+1]
            s[t+1] = H(u[t+1] - V_th)

        使用示例:
            s = lif.step(current)                    # full BPTT（默认）
            s = lif.step(current, detach_reset=True) # detached-reset
        """
        if detach_reset:
            # detached-reset：复位项从计算图 detach，反向 ∂u[t+1]/∂u[t] = λI（丢复位路径）
            self.u = self.lambd * self.u - self.lambd * self.v_th * self.s.detach() + current
        else:
            # full BPTT：复位项保留，反向 ∂u[t+1]/∂u[t] = λI - λV_th·D_Ψ
            self.u = self.lambd * (self.u - self.v_th * self.s) + current
        # 脉冲产生：前向 Heaviside（过阈值发 1），反向用 surrogate 梯度（SpikeFunction）
        self.s = SpikeFunction.apply(self.u, self.v_th, self.beta)
        return self.s

"""脉冲编码模块：把连续值输入编码成脉冲序列（阶段 0 任务 1 的一部分）。

SNN 的输入通常是脉冲序列。这里实现常用的泊松编码（rate encoding）：
把 [0,1] 的连续输入在每个时间步以概率 p = x 发放脉冲，时间平均的脉冲密度
等于输入值。
"""

import torch


def poisson_encode(x, T):
    """泊松编码：把 [0,1] 的连续输入编码成 T 个时间步的脉冲序列。

    参数:
        x (torch.Tensor): 连续输入，值在 [0,1]，形状 (batch, ...)。
        T (int): 时间步数。

    返回:
        torch.Tensor: 脉冲序列，形状 (T, batch, ...)，元素 0/1。

    原理:
        每个时间步、每个元素独立做伯努利采样：rand < x 则发 1，否则 0。
        期望发放率 E[s] = x，即时间平均的脉冲密度等于输入值。

    使用示例:
        spikes = poisson_encode(x, T=10)  # x: (batch, n_input) -> spikes: (T, batch, n_input)
    """
    # rand: (T, batch, ...)，每个时间步独立采样
    rand = torch.rand((T,) + tuple(x.shape), device=x.device)
    # x.unsqueeze(0): (1, batch, ...) 广播到 (T, batch, ...)，比较后转 float 得 0/1
    return (rand < x.unsqueeze(0)).float()

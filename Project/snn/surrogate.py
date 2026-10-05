"""surrogate 梯度函数（为后续反向传播预留，前向暂不使用）。

脉冲发放 s = H(u - V_th) 的导数几乎处处为 0，训练时用 surrogate 替代
∂s/∂u。这里实现 fast-sigmoid 形式（OTTT/NDOT 主线常用的选择）。

TODO（阶段 1）：当前仅定义函数，尚未接入 autograd。
"""

import torch


def fast_sigmoid_surrogate(u, v_th, beta=4.0):
    """计算 fast-sigmoid 形式的 surrogate 梯度。

    参数:
        u (torch.Tensor): 膜电位，任意形状。
        v_th (float): 发放阈值 V_th。
        beta (float): 锐度参数 β，默认 4.0。

    返回:
        torch.Tensor: surrogate 导数值，形状与 u 一致。

    对应公式:
        Ψ'(u - v_th) = 1 / (1 + β|u - v_th|)^2

    使用示例:
        g = fast_sigmoid_surrogate(u, V_TH)
    """
    # |u - v_th|：膜电位离阈值的距离；β 控制导数随距离增大的衰减速度
    return 1.0 / (1.0 + beta * (u - v_th).abs()) ** 2

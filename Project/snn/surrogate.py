"""surrogate 梯度函数（为后续反向传播预留，前向暂不使用）。

脉冲发放 s = H(u - V_th) 的导数几乎处处为 0，训练时用 surrogate 替代
∂s/∂u。这里实现 fast-sigmoid 形式（OTTT/NDOT 主线常用的选择）。
"""

import torch


def fast_sigmoid_surrogate(u, v_th, beta=4.0):
    """fast-sigmoid surrogate 梯度。

    Psi'(u - v_th) = 1 / (1 + beta * |u - v_th|)^2

    u: 膜电位（任意形状张量）
    v_th: 阈值
    beta: 锐度参数
    """
    return 1.0 / (1.0 + beta * (u - v_th).abs()) ** 2

"""surrogate 梯度函数与自定义 autograd 算子。

脉冲发放 s = H(u - V_th) 的导数几乎处处为 0，训练时用 surrogate 替代
∂s/∂u。这里实现 fast-sigmoid 形式（OTTT/NDOT 主线常用的选择）。

包含两个层次：
- fast_sigmoid_surrogate：纯函数，只算 Ψ'(u - v_th) 的值；
- SpikeFunction：自定义 autograd.Function，前向 Heaviside、反向用 surrogate，
  供 LIF 的 step 调用，让脉冲发放可导（阶段 0 任务）。
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


class SpikeFunction(torch.autograd.Function):
    """脉冲发放的自定义 autograd 算子：前向 Heaviside，反向 surrogate。

    前向: s = H(u - v_th) = (u >= v_th).float()
    反向: ∂L/∂u = (∂L/∂s) · Ψ'(u - v_th)，用 fast-sigmoid 替代不可导的 H'。

    这样 LIF 的脉冲发放就"可导"了：前向仍是硬阈值（二值脉冲），
    反向时梯度不为零，单样本前向 + 反向能贯通。
    """

    @staticmethod
    def forward(ctx, u, v_th, beta):
        """前向：Heaviside 硬阈值，产生二值脉冲（与不可导版本完全一致）。"""
        ctx.save_for_backward(u)  # 反向时要用的膜电位
        ctx.v_th = v_th           # 阈值（常数，不需求导）
        ctx.beta = beta           # 锐度参数（常数，不需求导）
        return (u >= v_th).float()

    @staticmethod
    def backward(ctx, grad_output):
        """反向：用 fast-sigmoid surrogate 替代 H'，即 ∂s/∂u ≈ Ψ'(u - v_th)。"""
        (u,) = ctx.saved_tensors
        # Ψ'(u - v_th) = 1 / (1 + β|u - v_th|)^2
        grad = 1.0 / (1.0 + ctx.beta * (u - ctx.v_th).abs()) ** 2
        # 链式法则：∂L/∂u = ∂L/∂s · ∂s/∂u；v_th、beta 是常数，返回 None
        return grad_output * grad, None, None

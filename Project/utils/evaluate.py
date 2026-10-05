"""指标接口（阶段 0 任务 4）。

提供 evaluate()，计算分类准确率，并预留逐层探针、CKA、梯度对齐的调用点（阶段 2）。
"""

import torch

from data.encoding import poisson_encode


def evaluate(model, x, y, T):
    """评估分类准确率。

    参数:
        model (MLPSNN): 多层 SNN。
        x (torch.Tensor): 连续输入，形状 (batch, n_input)。
        y (torch.Tensor): 标签，形状 (batch,)。
        T (int): 时间步数。

    返回:
        float: 分类准确率（0~1）。

    TODO（阶段 2）: 在这里预留逐层探针、CKA、梯度对齐的调用点，后续挂测量钩子。
    """
    model.eval()
    with torch.no_grad():
        spikes = poisson_encode(x, T)
        out = model(spikes)      # (T, batch, n_output)
        fr = out.mean(dim=0)     # firing rate 作 logits：(batch, n_output)
        pred = fr.argmax(dim=1)
        acc = (pred == y).float().mean().item()
    model.train()
    return acc

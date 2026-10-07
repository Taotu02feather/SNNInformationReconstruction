"""统一训练循环（阶段 0 任务 3）。

提供 train_epoch()，做「脉冲编码 -> 前向 -> 输出层 firing rate 损失 -> surrogate 反向 -> 更新」。
支持 accumulated（累积梯度，末尾更新）与 online（每步更新）两种模式骨架。
"""

import torch
import torch.nn.functional as F

from data.encoding import poisson_encode
from algorithm.bptt_sg import bptt_sg_step


def train_epoch(model, x, y, optimizer, T, mode="accumulated"):
    """训练一个 epoch（阶段 0 最简演示：firing rate 作 logits + 交叉熵）。

    参数:
        model (MLPSNN): 多层 SNN。
        x (torch.Tensor): 连续输入，形状 (batch, n_input)，值在 [0,1]。
        y (torch.Tensor): 标签，形状 (batch,)，类别索引（长整型）。
        optimizer (torch.optim.Optimizer): 优化器。
        T (int): 时间步数。
        mode (str): "accumulated"（累积，末尾更新，默认）或 "online"（每步更新）。

    返回:
        float: 本 epoch 的平均损失。

    说明:
        阶段 0 用输出层 firing rate（T 步脉冲的时间平均）作分类 logits、交叉熵作损失，
        验证「前向 + surrogate 反向 + 更新」的完整链路。
        accumulated / online 的区分在阶段 3（OTTT/NDOT 瞬时梯度）才真正体现；
        阶段 0 的 firing-rate 损失本质是累积式，online 与 accumulated 共用此逻辑，online 留 TODO。
    """
    # 1. 脉冲编码：连续输入 -> 脉冲序列 (T, batch, n_input)
    spikes = poisson_encode(x, T)
    # 2. 前向：输出层脉冲 (T, batch, n_output)
    out = model(spikes)
    # 3. 输出层 firing rate 作 logits：(batch, n_output)
    fr = out.mean(dim=0)
    # 4. 交叉熵损失
    loss = F.cross_entropy(fr, y)
    # 5. 反向 + 更新（surrogate 已接好，autograd 自动算梯度）
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def train_epoch_bptt_sg(model, x, y, optimizer, T, detach_reset=True, loss_mode="ce_fr"):
    """用手写 BPTT-SG 梯度训练一个 epoch。

    ⚠️ 仅阶段 1 sanity 用：验证手写梯度能驱动收敛。正式训练入口仍是 train_epoch（autograd full BPTT）。

    参数:
        model (MLPSNN): 多层 SNN。
        x (torch.Tensor): 连续输入 (batch, n_input)，值在 [0,1]。
        y (torch.Tensor): 标签 (batch,)。
        optimizer (torch.optim.Optimizer): 优化器。
        T (int): 时间步数。
        detach_reset (bool): True=丢复位项（A_t=λI，阶段 1 主线），False=full。
        loss_mode (str): "ce_fr" / "ce_sum_t" / "mse_fr"。

    返回:
        float: 本 epoch 损失。
    """
    spikes = poisson_encode(x, T)  # (T, batch, n_input)
    loss, grads = bptt_sg_step(model, spikes, y, detach_reset=detach_reset, loss_mode=loss_mode)
    # 手动填 grad（手写梯度，不经过 autograd）
    optimizer.zero_grad()
    for l in range(len(model.weights)):
        model.weights[l].grad = grads["dW"][l]
        model.biases[l].grad = grads["db"][l]
    optimizer.step()
    return loss

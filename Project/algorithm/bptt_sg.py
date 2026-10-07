"""BPTT-SG 手写前向 / 反向（阶段 1）。

符号对照（完整映射见 Project/SYMBOL_MAPPING.md，本文件正文只用代码记法）：
    论文 $W^{l\\leftarrow l-1}$（上标=目标层，1-based）  ↔  代码 `weights[l-1]`（下标=突触前层，0-based）
    论文 $u^l[t]$ ↔ 代码 `layer_u[l-1][t]`；论文 $\\delta^l[t]=\\partial L/\\partial u^l[t]$ ↔ 代码 `delta[l-1][t]`
    代码 `weights[l]` 连接第 $l$ 层→第 $l+1$ 层，形状 $(n_{l+1}, n_l)$；`layer_u[l]`/`layer_s[l]` 是第 $l+1$ 层序列。

前向（纯数值，torch.no_grad 下，不建计算图）：
    I^{l+1}[t] = W^l s^l[t] + b^l
    u^{l+1}[t] = λ(u^{l+1}[t-1] - V_th s^{l+1}[t-1]) + I^{l+1}[t]
    s^{l+1}[t] = H(u^{l+1}[t] - V_th)

反向（手写递推，detached-reset 主线 / full 对照）：
    输出层 L：δ^L[t] = dL_ds_out[t] ⊙ Ψ'(u^L[t]-V_th) + λ δ^L[t+1]      （full 再 -λV_th Ψ' ⊙ δ）
    中间层 l：δ^l[t] = ((W^{l+1})^T δ^{l+1}[t]) ⊙ Ψ'(u^l[t]-V_th) + λ δ^l[t+1]
    权重梯度：dW^l = Σ_t δ^l[t] ⊗ s^{l-1}[t]，db^l = Σ_t δ^l[t]
"""

import torch
import torch.nn.functional as F

from snn.surrogate import fast_sigmoid_surrogate


def _softmax(z):
    """数值稳定的 softmax（沿最后一维）。"""
    z = z - z.max(dim=-1, keepdim=True).values
    e = z.exp()
    return e / e.sum(dim=-1, keepdim=True)


def _compute_loss(s_out, y, loss_mode):
    """手写损失与输出层脉冲梯度 dL/ds_out（不依赖 autograd）。

    参数:
        s_out (torch.Tensor): 输出层脉冲序列，形状 (T, B, n_out)。
        y (torch.Tensor): 标签，形状 (B,)。
        loss_mode (str): "ce_fr"（firing rate + CE，默认）/ "ce_sum_t"（脉冲和 + CE）/ "mse_fr"（firing rate + MSE）。

    返回:
        (loss: float, dL_ds_out: (T, B, n_out))。dL_ds_out 每个 t 分量已含 /T 或 /1 因子。
    """
    T, B, n_out = s_out.shape
    onehot = F.one_hot(y, n_out).float()  # (B, n_out)

    if loss_mode == "ce_fr":
        fr = s_out.mean(dim=0)  # (B, n_out)
        p = _softmax(fr)
        loss = -(onehot * p.log()).sum(dim=-1).mean()
        grad_fr = (p - onehot) / B                 # ∂loss/∂fr
        dL_ds_out = grad_fr.unsqueeze(0).expand(T, B, n_out) / T  # ∂fr/∂s_out[t] = 1/T
    elif loss_mode == "ce_sum_t":
        logits = s_out.sum(dim=0)  # (B, n_out)
        p = _softmax(logits)
        loss = -(onehot * p.log()).sum(dim=-1).mean()
        grad_logits = (p - onehot) / B
        dL_ds_out = grad_logits.unsqueeze(0).expand(T, B, n_out)  # ∂logits/∂s_out[t] = 1
    elif loss_mode == "mse_fr":
        fr = s_out.mean(dim=0)
        diff = fr - onehot
        loss = (diff ** 2).mean()
        grad_fr = 2 * diff / (B * n_out)
        dL_ds_out = grad_fr.unsqueeze(0).expand(T, B, n_out) / T
    else:
        raise ValueError(f"未知 loss_mode: {loss_mode}（可选 ce_fr / ce_sum_t / mse_fr）")

    return loss.item(), dL_ds_out


def bptt_sg_forward(model, x):
    """纯数值前向（torch.no_grad 下，不建计算图），返回每层每步膜电位与脉冲。

    参数:
        model (MLPSNN): 多层 SNN。
        x (torch.Tensor): 输入脉冲序列，形状 (T, B, n_input)。

    返回:
        (layer_u, layer_s) 二元组，layer_u[l]/layer_s[l] 是第 l+1 层序列，形状 (T, B, n_{l+1})。
    """
    T, B, n_in = x.shape
    L = len(model.weights)  # 权重矩阵数 = 隐藏层数 + 输出层
    lambd = model.lif_layers[0].lambd
    v_th = model.lif_layers[0].v_th

    with torch.no_grad():
        layer_u = [[] for _ in range(L)]
        layer_s = [[] for _ in range(L)]
        # 初始状态 u[l][0] = 0, s[l][0] = 0
        u_prev = [torch.zeros(B, model.layer_sizes[l + 1], device=x.device) for l in range(L)]
        s_prev = [torch.zeros(B, model.layer_sizes[l + 1], device=x.device) for l in range(L)]

        for t in range(T):
            act = x[t]  # 第 0 层（输入层）脉冲 = x[t]，形状 (B, n_input)
            for l in range(L):
                current = act @ model.weights[l].T + model.biases[l]  # (B, n_{l+1})
                u = lambd * (u_prev[l] - v_th * s_prev[l]) + current  # 形式 A
                s = (u >= v_th).float()  # 前向 Heaviside（数值，detach 与否不影响前向）
                layer_u[l].append(u)
                layer_s[l].append(s)
                act = s
                u_prev[l] = u
                s_prev[l] = s

        layer_u = [torch.stack(lu, dim=0) for lu in layer_u]
        layer_s = [torch.stack(ls, dim=0) for ls in layer_s]
    return layer_u, layer_s


def bptt_sg_backward(model, layer_u, layer_s, dL_ds_out, x, detach_reset=True):
    """手写反向递推（纯数值），返回每层权重/偏置梯度。

    代码记法：delta[l] = ∂L/∂u^{l+1}[·]，形状 (T, B, n_{l+1})。
    空间反传：delta[l][t] = (delta[l+1][t] @ weights[l+1]) ⊙ Ψ'(u^{l+1}[t]-V_th)
    时间反传：delta[l][t] += λ delta[l][t+1]（detached-reset）；full 再减 λV_th Ψ' ⊙ delta[l][t+1]。

    参数:
        model (MLPSNN): 权重来源（只读 weights/biases）。
        layer_u / layer_s (list[Tensor]): bptt_sg_forward 的输出。
        dL_ds_out (torch.Tensor): 损失对输出层脉冲的梯度，形状 (T, B, n_out)。
        x (torch.Tensor): 输入脉冲序列 (T, B, n_input)，作为第 0 层 s（用于 dW[0]）。
        detach_reset (bool): True=丢复位项（A_t=λI），False=full（A_t=λI-λV_th D_Ψ）。

    返回:
        dict: {"dW": [dW[0..L-1]], "db": [db[0..L-1]]}，dW[l] 形状同 weights[l]，db[l] 形状同 biases[l]。
    """
    T, B, n_out = dL_ds_out.shape
    L = len(model.weights)
    lambd = model.lif_layers[0].lambd
    v_th = model.lif_layers[0].v_th
    beta = model.lif_layers[0].beta

    with torch.no_grad():
        # delta[l] 形状 (T, B, n_{l+1})
        delta = [torch.zeros(T, B, model.layer_sizes[l + 1], device=dL_ds_out.device) for l in range(L)]

        # 输出层 l = L-1
        l = L - 1
        for t in reversed(range(T)):
            psi = fast_sigmoid_surrogate(layer_u[l][t], v_th, beta)  # (B, n_out)
            d = dL_ds_out[t] * psi
            if t < T - 1:
                d = d + lambd * delta[l][t + 1]
                if not detach_reset:
                    d = d - lambd * v_th * psi * delta[l][t + 1]
            delta[l][t] = d

        # 中间层 l = L-2 .. 0
        for l in reversed(range(L - 1)):
            for t in reversed(range(T)):
                psi = fast_sigmoid_surrogate(layer_u[l][t], v_th, beta)  # (B, n_{l+1})
                # 空间反传：∂L/∂s^{l+1}[t] = delta[l+1][t] @ weights[l+1]
                d = (delta[l + 1][t] @ model.weights[l + 1]) * psi
                if t < T - 1:
                    d = d + lambd * delta[l][t + 1]
                    if not detach_reset:
                        d = d - lambd * v_th * psi * delta[l][t + 1]
                delta[l][t] = d

        # 权重/偏置梯度
        dW, db = [], []
        for l in range(L):
            s_prev = x if l == 0 else layer_s[l - 1]  # (T, B, n_l)，weights[l] 的输入
            dW_l = sum(delta[l][t].T @ s_prev[t] for t in range(T))  # (n_{l+1}, n_l)
            db_l = delta[l].sum(dim=(0, 1))  # (n_{l+1},)
            dW.append(dW_l)
            db.append(db_l)

    return {"dW": dW, "db": db}


def bptt_sg_step(model, spikes, y, detach_reset=True, loss_mode="ce_fr"):
    """前向 + 损失 + 手写反向一步（不含 optimizer.step）。

    参数:
        model (MLPSNN): 多层 SNN。
        spikes (torch.Tensor): 输入脉冲序列 (T, B, n_input)。
        y (torch.Tensor): 标签 (B,)。
        detach_reset (bool): True=丢复位项，False=full。
        loss_mode (str): "ce_fr" / "ce_sum_t" / "mse_fr"。

    返回:
        (loss: float, grads: dict)，grads 同 bptt_sg_backward 返回结构。
    """
    layer_u, layer_s = bptt_sg_forward(model, spikes)
    s_out = layer_s[-1]  # (T, B, n_out)
    loss, dL_ds_out = _compute_loss(s_out, y, loss_mode)
    grads = bptt_sg_backward(model, layer_u, layer_s, dL_ds_out, spikes, detach_reset=detach_reset)
    return loss, grads

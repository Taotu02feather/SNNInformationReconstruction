"""Gate A/B/C 梯度验证 + BPTT-SG 收敛曲线 sanity（阶段 1）。

符号对照（完整映射见 Project/SYMBOL_MAPPING.md）：论文 $W^{l\\leftarrow l-1}$ ↔ 代码 `weights[l-1]`；
论文 $\\delta^l[t]$ ↔ 代码 `delta[l-1][t]`。本文件正文只用代码记法。

- Gate A 验时间 Jacobian A_t = ∂u_t/∂u_{t-1}（detached=λI / full=λI-λV_th·D_Ψ，各一遍）。
- Gate B 验空间 Jacobian B = ∂u^l/∂u^{l-1} = W·D_Ψ（单层单步）。
- Gate C 验最终权重/偏置梯度 dW/db（多层多步，detached 与 full 各一遍）。
- train_sanity 验手写 BPTT-SG 梯度能驱动合成数据收敛。

运行：python -m sanity.gradient_check
"""

import torch
import torch.nn.functional as F

from algorithm.bptt_sg import bptt_sg_step
from data.dataset import synthetic_dataset
from network.mlp_snn import MLPSNN
from snn.surrogate import SpikeFunction, fast_sigmoid_surrogate
from utils.config import LAMBDA, V_TH, BETA
from utils.evaluate import evaluate
from utils.train import train_epoch_bptt_sg


def rel_err(a, b):
    """全局相对误差：max|a-b| / max(|b|, 1e-6)。返回标量 float。"""
    denom = b.abs().max().clamp_min(1e-6)
    return ((a - b).abs().max() / denom).item()


def assert_close(a, b, name=""):
    e = rel_err(a, b)
    ok = torch.allclose(a, b, rtol=1e-5, atol=1e-8)
    print(f"  {name}: max rel_err={e:.3e}, allclose={ok}")
    assert ok and e < 1e-5, f"{name} 梯度不一致: rel_err={e:.3e}"
    return e


def _transfer(u_prev, I_t, lambd, v_th, beta, detach_reset):
    """单步时间转移：u_t = λ u_prev - λ V_th s_prev + I_t，s_prev = H(u_prev - V_th)。"""
    s_prev = SpikeFunction.apply(u_prev, v_th, beta)
    if detach_reset:
        return lambd * u_prev - lambd * v_th * s_prev.detach() + I_t
    return lambd * u_prev - lambd * v_th * s_prev + I_t


def gate_A(n=4, T=5):
    """Gate A：时间 Jacobian A_t = ∂u_t/∂u_{t-1}（detached 与 full 各一遍）。"""
    torch.manual_seed(0)
    lambd, v_th, beta = LAMBDA, V_TH, BETA
    I = torch.randn(T, n) * 0.5

    # 手写数值前向（不建图），u_list[k] = u_k（k = 0..T）
    u_prev = torch.zeros(n)
    s_prev = torch.zeros(n)
    u_list = [u_prev.clone()]
    for t in range(T):
        u_t = lambd * (u_prev - v_th * s_prev) + I[t]
        s_t = (u_t >= v_th).float()
        u_list.append(u_t.clone())
        u_prev, s_prev = u_t, s_t

    for mode, detach_reset in [("detached", True), ("full", False)]:
        ok = True
        max_err = 0.0
        for t in range(1, T + 1):
            u_prev_val = u_list[t - 1]  # u_{t-1}
            psi = fast_sigmoid_surrogate(u_prev_val, v_th, beta)  # (n,)
            if detach_reset:
                A_manual = lambd * torch.eye(n)
            else:
                A_manual = lambd * torch.eye(n) - lambd * v_th * torch.diag(psi)
            u_leaf = u_prev_val.clone().requires_grad_(True)
            A_ad = torch.autograd.functional.jacobian(
                lambda up: _transfer(up, I[t - 1], lambd, v_th, beta, detach_reset), u_leaf
            )
            e = rel_err(A_manual, A_ad)
            max_err = max(max_err, e)
            if e >= 1e-5:
                ok = False
        print(f"  Gate A ({mode}): max rel_err={max_err:.3e} -> {'PASS' if ok else 'FAIL'}")
        assert ok, f"Gate A ({mode}) 未通过"


def _spatial_transfer(u_in, W, b, u_self_prev, s_self_prev, lambd, v_th, beta):
    """空间转移：u^l = λ(u_self_prev - v_th s_self_prev) + W s_in + b，s_in = H(u_in - v_th)。"""
    s_in = SpikeFunction.apply(u_in, v_th, beta)
    return lambd * (u_self_prev - v_th * s_self_prev) + W @ s_in + b


def gate_B(n_in=4, n_out=5):
    """Gate B：空间 Jacobian B = ∂u^l/∂u^{l-1} = W·D_Ψ（单层单步）。"""
    torch.manual_seed(1)
    W = torch.randn(n_out, n_in) * 0.5
    b = torch.zeros(n_out)
    u_self_prev = torch.randn(n_out) * 0.3
    s_self_prev = (u_self_prev >= V_TH).float()
    u_in_val = torch.randn(n_in) * 0.5  # u^{l-1}[t] 数值

    psi = fast_sigmoid_surrogate(u_in_val, V_TH, BETA)  # (n_in,)
    B_manual = W * psi.unsqueeze(0)  # (n_out, n_in)

    u_in_leaf = u_in_val.clone().requires_grad_(True)
    B_ad = torch.autograd.functional.jacobian(
        lambda uin: _spatial_transfer(uin, W, b, u_self_prev, s_self_prev, LAMBDA, V_TH, BETA),
        u_in_leaf,
    )  # (n_out, n_in)

    assert_close(B_manual, B_ad, "Gate B")


def gate_C(layer_sizes=(8, 16, 3), B=4, T=10):
    """Gate C：最终权重/偏置梯度 dW/db（多层多步，detached 与 full 各一遍）。"""
    for mode, detach_reset in [("detached", True), ("full", False)]:
        torch.manual_seed(2)
        model = MLPSNN(list(layer_sizes), LAMBDA, V_TH)
        spikes = (torch.rand(T, B, layer_sizes[0]) < 0.5).float()
        y = torch.randint(0, layer_sizes[-1], (B,))

        # 手写：bptt_sg_step 内部做 数值前向 + 手写损失 + 手写反向
        _, grads = bptt_sg_step(model, spikes, y, detach_reset=detach_reset, loss_mode="ce_fr")

        # autograd 参考：detached-reset 前向 + CE 损失 + backward
        out = model(spikes, detach_reset=detach_reset)  # (T, B, n_out)
        fr = out.mean(dim=0)
        loss = F.cross_entropy(fr, y)
        model.zero_grad()
        loss.backward()

        ok = True
        for l in range(len(model.weights)):
            eW = rel_err(grads["dW"][l], model.weights[l].grad)
            eb = rel_err(grads["db"][l], model.biases[l].grad)
            print(f"    layer {l}: rel_err(W)={eW:.3e}, rel_err(b)={eb:.3e}")
            if eW >= 1e-5 or eb >= 1e-5:
                ok = False
        print(f"  Gate C ({mode}): {'PASS' if ok else 'FAIL'}")
        assert ok, f"Gate C ({mode}) 未通过"


def train_sanity():
    """收敛曲线：手写 BPTT-SG 梯度训练合成数据，断言收敛且稳定。

    用 full BPTT（detach_reset=False）验证「跑穿上限」：完整梯度（含 reset 项）应能到 0.85+，
    证明手写反向的完整结构正确且能训练。
    detached-reset（丢 reset 项，A_t=λI）的收敛上限更低，是丢复位项的系统性近似代价，
    作为对照打印（阶段 3 对比 OTTT 丢 reset 项时参照），不作收敛断言。

    注：合成数据取 256 样本（64 样本 + 0.2 噪声 + 泊松编码的可分性不足，full 也难稳定到 0.85）。
    """
    torch.manual_seed(3)
    n_input, n_hidden, n_output = 8, 64, 3
    n_samples, T = 256, 10
    x, y = synthetic_dataset(n_samples, n_input, n_output)

    for detach_reset, tag in [(False, "full"), (True, "detached")]:
        model = MLPSNN([n_input, n_hidden, n_output], LAMBDA, V_TH)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        accs = []
        for epoch in range(80):
            loss = train_epoch_bptt_sg(model, x, y, optimizer, T, detach_reset=detach_reset, loss_mode="ce_fr")
            accs.append(evaluate(model, x, y, T))

        last3 = accs[-3:]
        mean_acc = sum(last3) / 3
        var_acc = sum((a - mean_acc) ** 2 for a in last3) / 3
        print(f"  [{tag}] 最后3 epoch acc={[f'{a:.3f}' for a in last3]}, 均值={mean_acc:.3f}, 方差={var_acc:.3e}")

        if not detach_reset:  # full 是「上限」，作断言
            assert mean_acc > 0.85, f"full BPTT 未收敛到 0.85（均值 {mean_acc:.3f}），梯度可能有问题"
            assert var_acc < 0.01, f"full BPTT 最后 3 epoch 方差过大（{var_acc:.3e}），训练不稳定"
    print("  train_sanity: PASS")


def main():
    print("== Gate A（时间 Jacobian A_t）==")
    gate_A()
    print("== Gate B（空间 Jacobian B）==")
    gate_B()
    print("== Gate C（最终梯度 dW/db）==")
    gate_C()
    print("== 收敛曲线（手写 BPTT-SG）==")
    train_sanity()
    print("\nPASS: 阶段 1 全部通过（Gate A/B/C + 收敛曲线）")


if __name__ == "__main__":
    main()

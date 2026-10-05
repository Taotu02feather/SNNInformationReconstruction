"""前向验证：手算一个 n=2, T=3 的 LIF 例子，与代码对照。

运行：python -m sanity.forward_check
"""

import torch

from network.mlp_snn import MLPSNN
from snn.lif import LIF
from snn.surrogate import SpikeFunction
from utils.config import LAMBDA, V_TH, BETA


def main():
    """手算 n=2、T=3 的 LIF 前向例子，与代码前向对照；再检查 MLPSNN 输出形状。

    手算过程（lambda=0.5, V_th=1.0）：
        u[t+1] = 0.5 * (u[t] - 1.0 * s[t]) + I[t+1]
        s[t+1] = H(u[t+1] - 1.0)

    返回:
        无（打印结果；不一致时 raise SystemExit(1)）。

    运行:
        python -m sanity.forward_check
    """
    n = 2  # 神经元数
    T = 3  # 时间步数

    # 输入电流序列（每个时刻一个 (n,) 向量），整体形状 (T, n)
    inputs = torch.tensor([
        [0.5, 1.5],   # I[1]
        [1.0, 0.5],   # I[2]
        [0.25, 1.0],  # I[3]
    ])

    # 手算期望值（lambda=0.5, V_th=1.0），用于对照代码输出
    expected_u = torch.tensor([
        [0.5, 1.5],      # u[1] = 0.5*(0 - 1.0*0) + [0.5, 1.5]
        [1.25, 0.75],    # u[2] = 0.5*([0.5,1.5] - 1.0*[0,1]) + [1.0, 0.5]
        [0.375, 1.375],  # u[3] = 0.5*([1.25,0.75] - 1.0*[1,0]) + [0.25, 1.0]
    ])
    expected_s = torch.tensor([
        [0.0, 1.0],  # s[1] = H(u[1] - 1.0)
        [1.0, 0.0],  # s[2] = H(u[2] - 1.0)
        [0.0, 1.0],  # s[3] = H(u[3] - 1.0)
    ])

    # 代码前向：用 LIF 实例跑 T 个时间步
    lif = LIF(LAMBDA, V_TH)
    lif.reset((n,))  # 初始化 u[0]=0, s[0]=0
    u_list, s_list = [], []
    for t in range(T):
        s = lif.step(inputs[t])       # 单步前向，输入 I[t+1]
        u_list.append(lif.u.clone())  # 记录本步膜电位
        s_list.append(s.clone())      # 记录本步脉冲

    # 堆叠成 (T, n)
    u_code = torch.stack(u_list)
    s_code = torch.stack(s_list)

    # 用 allclose 判断代码输出与手算期望值是否一致（允许浮点误差）
    u_ok = torch.allclose(u_code, expected_u, atol=1e-6)
    s_ok = torch.allclose(s_code, expected_s, atol=1e-6)

    print("== LIF 前向手算验证 ==")
    print("膜电位 u[t] =")
    print(u_code)
    print("脉冲 s[t] =")
    print(s_code)
    print(f"膜电位一致: {u_ok}")
    print(f"脉冲一致: {s_ok}")

    if u_ok and s_ok:
        print("PASS: 代码前向与手算一致。")
    else:
        print("FAIL: 代码前向与手算不一致！")
        raise SystemExit(1)

    # 顺带验证 MLPSNN 输出形状：(T, batch, n_input) -> (T, batch, n_output)
    model = MLPSNN([2, 4, 2], LAMBDA, V_TH)
    x = torch.randn(T, 1, 2)
    out = model(x)
    print("\n== MLPSNN 形状检查 ==")
    print(f"输入形状: {tuple(x.shape)} -> 输出形状: {tuple(out.shape)}")
    assert out.shape == (T, 1, 2), "MLPSNN 输出形状错误"
    print("PASS: MLPSNN 输出形状正确。")

    # 梯度检查：验证 surrogate 反向贯通（阶段 0 第 5 条任务）
    print("\n== surrogate 反向贯通检查 ==")
    # 两个神经元：0.5 离阈值远、1.0 恰在阈值上
    u_test = torch.tensor([0.5, 1.0], requires_grad=True)
    s_test = SpikeFunction.apply(u_test, V_TH, BETA)
    s_test.sum().backward()  # 以脉冲之和作损失，反向传播
    print(f"脉冲 s = {s_test.detach().tolist()}")
    print(f"dL/du = {u_test.grad.tolist()}")
    # 反向应产生非零梯度（surrogate 生效），且离阈值近的（1.0）梯度更大
    assert u_test.grad is not None and torch.any(u_test.grad != 0), "surrogate 反向未贯通"
    print("PASS: surrogate 反向能产生梯度（单样本前向 + 反向贯通）。")


if __name__ == "__main__":
    main()

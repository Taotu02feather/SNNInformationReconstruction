"""前向验证：手算一个 n=2, T=3 的 LIF 例子，与代码对照。

运行：python -m sanity.forward_check
"""

import torch

from network.mlp_snn import MLPSNN
from snn.lif import LIF
from utils.config import LAMBDA, V_TH


def main():
    n = 2
    T = 3

    # 输入电流序列（每个时刻一个 (n,) 向量）
    inputs = torch.tensor([
        [0.5, 1.5],   # I[1]
        [1.0, 0.5],   # I[2]
        [0.25, 1.0],  # I[3]
    ])

    # 手算期望值（lambda=0.5, V_th=1.0）
    expected_u = torch.tensor([
        [0.5, 1.5],      # u[1]
        [1.25, 0.75],    # u[2]
        [0.375, 1.375],  # u[3]
    ])
    expected_s = torch.tensor([
        [0.0, 1.0],  # s[1]
        [1.0, 0.0],  # s[2]
        [0.0, 1.0],  # s[3]
    ])

    # 代码前向
    lif = LIF(LAMBDA, V_TH)
    lif.reset((n,))
    u_list, s_list = [], []
    for t in range(T):
        s = lif.step(inputs[t])
        u_list.append(lif.u.clone())
        s_list.append(s.clone())

    u_code = torch.stack(u_list)
    s_code = torch.stack(s_list)

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

    # 顺带验证 MLPSNN 输出形状
    model = MLPSNN([2, 4, 2], LAMBDA, V_TH)
    x = torch.randn(T, 1, 2)
    out = model(x)
    print("\n== MLPSNN 形状检查 ==")
    print(f"输入形状: {tuple(x.shape)} -> 输出形状: {tuple(out.shape)}")
    assert out.shape == (T, 1, 2), "MLPSNN 输出形状错误"
    print("PASS: MLPSNN 输出形状正确。")


if __name__ == "__main__":
    main()

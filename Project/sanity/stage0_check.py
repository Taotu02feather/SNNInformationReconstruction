"""阶段 0 框架验证：数据编码 -> 网络前向/反向 -> 训练循环 -> 指标接口 -> 逐层状态/参数顺序。

运行：python -m sanity.stage0_check
"""

import torch

from data.dataset import synthetic_dataset
from data.encoding import poisson_encode
from network.mlp_snn import MLPSNN
from utils.evaluate import evaluate
from utils.train import train_epoch


def main():
    torch.manual_seed(0)
    n_input, n_hidden, n_output = 8, 16, 3
    n_samples, T = 64, 10

    # 合成数据（阶段 0 用合成数据演示，真实 DVS 数据留 TODO）
    x, y = synthetic_dataset(n_samples, n_input, n_output)
    model = MLPSNN([n_input, n_hidden, n_output])
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-2)

    print("== 阶段 0 框架验证 ==")
    for epoch in range(20):
        loss = train_epoch(model, x, y, optimizer, T)
        if epoch % 5 == 0 or epoch == 19:
            acc = evaluate(model, x, y, T)
            print(f"epoch {epoch}: loss={loss:.4f}, acc={acc:.3f}")

    acc = evaluate(model, x, y, T)
    assert acc > 0.6, f"准确率过低: {acc:.3f}（应能学到可分合成数据）"

    # 验证 forward_states 返回每层 u, s
    spikes = poisson_encode(x[:4], T)
    out, layer_u, layer_s = model.forward_states(spikes)
    assert out.shape == (T, 4, n_output), f"输出形状错误: {out.shape}"
    assert len(layer_u) == len(layer_s) == 2, "应返回 2 层（隐藏 + 输出）的状态"
    print(f"\n输出脉冲形状: {tuple(out.shape)}")
    for l, (u, s) in enumerate(zip(layer_u, layer_s)):
        print(f"第 {l+1} 层 u 形状 {tuple(u.shape)}, s 形状 {tuple(s.shape)}")

    # 验证 param_vector 固定参数顺序
    pv = model.param_vector()
    assert pv.ndim == 1
    print(f"\n参数向量形状: {tuple(pv.shape)}")

    print("\nPASS: 阶段 0 框架跑通。")


if __name__ == "__main__":
    main()

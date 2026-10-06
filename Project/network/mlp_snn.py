"""最简单的多层 SNN（仅前向，不做学习）。

TODO（阶段 0）：当前 forward 只返回输出层脉冲，缺"返回每层 u, s"与"固定参数顺序"。
"""

import torch
import torch.nn as nn

from snn.lif import LIF
from utils.config import LAMBDA, V_TH


class MLPSNN(nn.Module):
    """多层 LIF 前馈 SNN。

    结构由 layer_sizes 决定，例如 [n_input, n_hidden, n_output]。
    输入层直接接收外部输入，其余每层是一个 LIF 神经元层。

    层间连接（线性）：
        I^{l+1}[t] = W^{l+1←l} s^l[t] + b^{l+1}
    每层内部按 LIF 离散更新（见 snn/lif.py）。

    属性：
        lif_layers: 除输入层外每层一个 LIF 实例（nn.ModuleList）
        weights: 层间权重，weights[l] 形状 (n_out, n_in)（nn.ParameterList）
        biases: 层间偏置，biases[l] 形状 (n_out,)（nn.ParameterList）
    """

    def __init__(self, layer_sizes, lambd: float = LAMBDA, v_th: float = V_TH):
        """按 layer_sizes 构造多层 LIF 前馈 SNN。

        参数:
            layer_sizes (list[int]): 各层神经元数，如 [n_input, n_hidden, n_output]。
            lambd (float): 膜衰减因子 λ，默认 config.LAMBDA=0.5。
            v_th (float): 发放阈值 V_th，默认 config.V_TH=1.0。

        返回:
            无（构造实例）。

        使用示例:
            model = MLPSNN([2, 4, 2])
        """
        super().__init__()
        self.layer_sizes = list(layer_sizes)
        # 除输入层（第 0 层）外，每一层各有一个 LIF 神经元层
        self.lif_layers = nn.ModuleList(
            [LIF(lambd, v_th) for _ in self.layer_sizes[1:]]
        )
        # 层间全连接权重与偏置：weights[l] 连接第 l 层 → 第 l+1 层
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()
        for n_in, n_out in zip(self.layer_sizes[:-1], self.layer_sizes[1:]):
            # 权重形状 (n_out, n_in)，随机初始化 *0.5：保证膜电位能达到阈值（否则神经元"死"、不发脉冲）
            self.weights.append(nn.Parameter(torch.randn(n_out, n_in) * 0.5))
            self.biases.append(nn.Parameter(torch.zeros(n_out)))

    def forward(self, x):
        """逐时间步做多层前向。

        参数:
            x (torch.Tensor): 外部输入序列，形状 (T, batch_size, n_input)。

        返回:
            torch.Tensor: 输出层脉冲序列，形状 (T, batch_size, n_output)。

        使用示例:
            out = model(x)  # x: (T, B, n_input)

        TODO（阶段 0）: 当前只返回输出层脉冲，缺"返回每层 u, s"（供探针/CKA 使用）
            与"固定参数顺序"（供梯度对齐展平使用）。
        """
        T, batch, _ = x.shape
        n_layers = len(self.layer_sizes)

        # 初始化每层状态：u^l[0] = 0, s^l[0] = 0
        for l in range(n_layers - 1):
            self.lif_layers[l].reset((batch, self.layer_sizes[l + 1]))

        outputs = []
        for t in range(T):
            act = x[t]  # 输入层（第 0 层）活动即外部输入，形状 (batch, n_input)
            for l in range(n_layers - 1):
                # 线性变换得到第 l+1 层的输入电流 I^{l+1}[t] = W s^l[t] + b
                current = act @ self.weights[l].T + self.biases[l]
                # 该层 LIF 单步更新，输出其脉冲作为下一层的输入
                act = self.lif_layers[l].step(current)
            outputs.append(act)  # 记录当前时间步的输出层脉冲
        # 把 T 个时间步的输出堆叠成 (T, batch, n_output)
        return torch.stack(outputs, dim=0)

    def forward_states(self, x):
        """前向传播，并返回每层的膜电位 u 与脉冲 s。

        参数:
            x (torch.Tensor): 外部输入序列，形状 (T, batch_size, n_input)。

        返回:
            (out, layer_u, layer_s) 三元组：
            - out: (T, batch_size, n_output) 输出层脉冲序列；
            - layer_u: list[Tensor]，layer_u[l] 是第 l+1 层（即 lif_layers[l]）的膜电位序列，
              形状 (T, batch_size, n_{l+1})；
            - layer_s: list[Tensor]，layer_s[l] 是第 l+1 层的脉冲序列，形状同上。

        用途:
            供阶段 2 的逐层探针 / CKA 使用（需要每层的表示）。
        """
        T, batch, _ = x.shape
        n_layers = len(self.layer_sizes)

        # 初始化每层状态：u^l[0] = 0, s^l[0] = 0
        for l in range(n_layers - 1):
            self.lif_layers[l].reset((batch, self.layer_sizes[l + 1]))

        layer_u = [[] for _ in range(n_layers - 1)]
        layer_s = [[] for _ in range(n_layers - 1)]
        outputs = []
        for t in range(T):
            act = x[t]  # 输入层活动即外部输入
            for l in range(n_layers - 1):
                current = act @ self.weights[l].T + self.biases[l]
                act = self.lif_layers[l].step(current)
                layer_u[l].append(self.lif_layers[l].u.clone())  # 记录第 l+1 层本步膜电位
                layer_s[l].append(act.clone())                   # 记录第 l+1 层本步脉冲
            outputs.append(act)
        # 各层状态堆叠成 (T, batch, n) 形式
        layer_u = [torch.stack(lu, dim=0) for lu in layer_u]
        layer_s = [torch.stack(ls, dim=0) for ls in layer_s]
        return torch.stack(outputs, dim=0), layer_u, layer_s

    def param_vector(self):
        """按"层交替"顺序展平所有参数为 1D 向量。

        返回:
            torch.Tensor: 形状 (num_params,) 的 1D 张量。

        顺序（层交替，行优先展平）:
            [W_0, b_0, W_1, b_1, ..., W_{L-1}, b_{L-1}]
            其中 W_l = self.weights[l]（第 l 层权重），b_l = self.biases[l]（第 l 层偏置），
            每个参数用 reshape(-1) 行优先展平。
            按层交替可让梯度对齐阶段直接"按层切片"（每层占 W_l.numel()+b_l.numel() 个元素）。

        用途:
            供阶段 2 的逐层梯度对齐使用（需按层切片梯度）。

        断言:
            - weights 与 biases 层数一致（防止未来改动破坏层序）；
            - 展平总长度 = 所有参数 numel 之和（防止遗漏某层）。
        """
        assert len(self.weights) == len(self.biases), \
            f"weights 与 biases 层数不一致: {len(self.weights)} vs {len(self.biases)}"
        vecs = []
        for l in range(len(self.weights)):
            vecs.append(self.weights[l].reshape(-1))  # W_l 行优先展平
            vecs.append(self.biases[l].reshape(-1))   # b_l 展平
        total = sum(w.numel() + b.numel() for w, b in zip(self.weights, self.biases))
        out = torch.cat(vecs)
        assert out.numel() == total, f"展平长度 {out.numel()} 与期望 {total} 不一致"
        return out

import torch
from torch import nn
import torch.optim as optim
from torch.nn import functional as F

__all__ = ["LocalLearningSignalGenerationLayer"]

'''
这个文件定义了一个模块 LocalLearningSignalGenerationLayer。

它把任意一个网络块 block 包起来，在训练时不使用全局反向传播误差，而是：

- 先算出这个 block 的输出 out

- 从 out 中池化出一个低维表示 latents

- 用一组固定的频率基/波形基 basis,把 latents 投影成类别得分 layer_pred

- 用标签直接在这个 block 输出上构造一个局部分类损失

- 对这个局部损失调用 loss.backward()

- 只更新这个 block 内部参数，而不把梯度继续传给更前面的层


这和论文中“每层自己生成 learning signal，而不是从最终层反向传播误差”的思想是一致的,也就是这里是m^(l)[t]的部分




'''
class LinearSigmoid(torch.autograd.Function):
    """
    Surrogate gradient based on arctan, used in Feng et al. (2021)
    这在代码中只用于 loss_function == "MSEHW" 的情况：
    它的作用是把每个类别分数压到近似 [0,1] 区间，再做 one-hot 回归。这个分支不是论文 TESS 的主公式，但属于一种局部监督替代实现。
    """
    @staticmethod
    def forward(ctx, x):
        result = torch.zeros_like(x)
        # Segment 1: x <= -2 -> approximate with 0
        result = torch.where(x <= -2, torch.zeros_like(x), result)
        # Segment 2: -2 < x < 2 -> approximate with 0.25 * x + 0.5
        result = torch.where((x > -2) & (x < 2), 0.25 * x + 0.5, result)
        # Segment 3: x >= 2 -> approximate with 1
        result = torch.where(x >= 2, torch.ones_like(x), result)
        return result

    @staticmethod
    def backward(ctx, grad_output):

        return grad_output, None


def generate_frequency_matrix(num_rows, num_cols, min_freq=50, max_freq=2000, freq=None):
    """
    这个函数生成一个固定基矩阵:

    既支持连续正弦基，也支持论文更贴近的二值方波基

    投影矩阵 B(l) 的设计是 LSG 流程的一个关键组成部分。 在 TESS 中， B(l) 被定义为一个固定的二进制矩阵，每 一列对应一个方波函数。这种设计具有多个优势。方波函 数通过为不同类别分配不同的空间频率，帮助同步同一层 内神经元的活性，确保与任务相关的信息能够有效地分布 在层中。此外， B(l) 的列是 近似正交 的，最小化了不同 类别投影之间的干扰。方波函数的简单性也使其具有很高 的硬件效率，这在资源受限的环境中尤其有利。


    """
    if freq is None:
        frequencies = torch.linspace(min_freq, max_freq, num_rows).unsqueeze(1).cuda()
    else:
        frequencies = freq
    # phases = torch.randn(num_rows, 1) * 2 * 3.14159
    t = torch.arange(num_cols).float().unsqueeze(0).cuda()
    sinusoids = torch.sin(frequencies * t )
    return sinusoids
    
'''
固定基矩阵用于计算后续内容固定矩阵
'''


def compute_LLS(activation, labels, temperature=1, label_smoothing=0.0, act_size=1, n_classes=10,
                modulation_term=None, modulation=False, freq=None, waveform="cosine", loss_function="CE"):
    """
    计算本地学习信号Local Learning Signal

    对应论文中固定矩阵: B(l) 的设计:

    论文说：
        - B(l)是固定二值矩阵
        - 列对应 square wave functions
        - 其作用是让不同类别对应不同空间频率，并尽量 quasi-orthogonal

    代码对应：

        - generate_frequency_matrix 生成按频率排布的波形基
        - waveform=="square" 时使用 sign(sin()) 变成方波
        - basis 在训练中不更新，是固定的

    所以代码里的 basis 就是论文里的 B(l)的实现版本。
    
    不同之处仅在于： 代码允许正弦/方波两种;论文更强调二值 square-wave



    """
    batch_size = activation.size(0)
    if activation.dim() == 4:
        latents = F.adaptive_avg_pool2d(activation, (act_size, act_size)).view(batch_size, -1)
    else:
        latents = F.adaptive_avg_pool1d(activation, act_size).view(batch_size, -1)
    basis = generate_frequency_matrix(n_classes, latents.size(1), max_freq=512, freq=freq).cuda()
    ## 把 block 输出变成局部特征 latents: 为了让任意 block 的输出都能映射到一个固定维度，再接局部监督。

    # basis = generate_frequency_matrix(n_classes, latents.size(1), max_freq=latents.size(1) - 50).cuda()
    if waveform == "square":
        basis = torch.sign(basis)
    basis = basis/latents.size(1)
    ## 生成固定基矩阵 basis
    # latents = F.normalize(latents, dim=1)
   
    layer_pred = torch.matmul(latents, basis.T)
    ## layer_pred = latents @ basis.T
    ## 这就是“把该层特征投影到类别子空间”的过程;只不过论文用的是时刻 t 的 spike 向量 o(l)[t]，而代码这里用的是池化后的连续特征 latents

    if modulation == 1:
        layer_pred = modulation_term*layer_pred
    if modulation == 2:
        layer_pred = torch.matmul(layer_pred, modulation_term)

    if loss_function == "CE":
        loss = torch.nn.functional.cross_entropy(layer_pred / temperature, labels, label_smoothing=label_smoothing)
    ## loss = CE(s/T, y)
    elif loss_function == "MSEHW":
        loss = torch.nn.functional.mse_loss(LinearSigmoid.apply(layer_pred / temperature),
                                            torch.nn.functional.one_hot(labels, num_classes=n_classes).float())
    else:
        raise NotImplementedError(f"{loss_function} is not implemented")
    return loss
    ## 这里就是开始计算局部损失了，使用了CE交叉熵和MSEHW线性sigmoid两种方法 
    ## 论文显式写出本地 learning signal；代码把它写成一个局部分类损失，由 autograd 隐式求出同一个 learning signal。


class LocalLearningSignalGenerationLayer(nn.Module):
    """
    分为推理/标准 BP 模式 与 LLS 模式
    """
    def __init__(self, block:nn.Module, lr=1e-1, n_classes=10, momentum=0, weight_decay=0,
                 nesterov=False, optimizer="SGD", milestones=[10, 30, 50], gamma=0.1, training_mode="LLS",
                 lr_scheduler = "MultiStepLR", patience=20, temperature=1, label_smoothing=0.0, dropout=0.0,
                 waveform="cosine", hidden_dim = 2048, reduced_set=20, pooling_size = 4, scaler = False,
                 cosine_lr=200, loss_function="CE"):
        super(LocalLearningSignalGenerationLayer, self).__init__()
        self.block = block
        self.lr = lr
        self.n_classes = n_classes
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        self.training_mode = training_mode
        self.patience = patience
        self.temperature = temperature
        self.label_smoothing = label_smoothing
        self.dropout = dropout
        self.waveform = waveform
        self.milestones = milestones
        self.gamma = gamma
        self.hidden_dim = hidden_dim
        self.reduced_set = reduced_set
        self.pooling_size = pooling_size
        self.scaler = None
        self.loss_function = loss_function

        if optimizer == "SGD":
            self.optimizer = optim.SGD(self.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay,
                                       nesterov=nesterov)
        elif optimizer == "Adam":
            self.optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"{optimizer} is not supported")

        if lr_scheduler == "MultiStepLR":
            self.lr_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, gamma=gamma, milestones=milestones)
        elif lr_scheduler == "ReduceLROnPlateau":
            self.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=gamma, patience=patience)
        elif lr_scheduler == "CosineLR":
            self.lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, cosine_lr)

        self.loss_hist = 0
        self.samples = 0
        self.loss_avg = 0

    def record_statistics(self, loss, batch_size):
        self.loss_hist += loss.item() * batch_size
        self.samples += batch_size
        self.loss_avg = self.loss_hist / self.samples if self.samples > 0 else 0

    def reset_statistics(self):
        self.loss_hist = 0
        self.samples = 0
        self.loss_avg = 0

    def optimizer_zero_grad(self):
        if hasattr(self, "optimizer"):
            self.optimizer.zero_grad()

    def optimizer_step(self):
        if hasattr(self, "optimizer"):
            self.optimizer.step()

    def forward(self, x, labels=None, feedback=None, x_err=None):
        training = self.training

        if self.training_mode == "BP" or not training or labels is None:
            return self.block(x)
        else:
            out = self.block(x.detach())

            if self.training_mode == "LLS":
                temperature = self.temperature
                label_smoothing = self.label_smoothing
                loss = compute_LLS(out, labels, temperature, label_smoothing, self.pooling_size,
                                   self.n_classes, waveform=self.waveform, loss_function=self.loss_function)

            else:
                raise NotImplementedError(f"Unknown training mode: {self.training_mode}")

            # self.optimizer.zero_grad()
            loss.backward()
            # self.optimizer.step()
            self.record_statistics(loss.detach(), x.size(0))

            ## x.detach() 断开与前层图连接
            ## return out.detach() 断开与后层图连接
            ## 当前 block 只由自己的局部 loss 更新

            return out.detach()


"""
Summary: 

代码中的 basis 对应论文中的固定投影矩阵 B(l)

代码中的 layer_pred = latents @ basis.T 对应论文中 B(l)o(l)[t] 的前向投影。

代码中的 cross_entropy(layer_pred / temperature, labels) 在反向传播时产生的误差 与论文 Eq. (9) 的m=B^T(f(Bo-y))本质对应

x.detach() 和 out.detach() 使得每个 block 只受自身局部损失训练，体现了论文强调的空间局部性。


"""
"""数据集接口（阶段 0 任务 1 的一部分）。

真实数据集（DVS Gesture、CIFAR10-DVS）需要下载，这里先提供合成数据生成器，
用于阶段 0 的框架验证。真实数据加载留 TODO。
"""

import torch


def synthetic_dataset(n_samples, n_input, n_classes, seed=0):
    """生成可分的合成数据集：每类一个随机模板，输入 = 模板 + 噪声。

    输入和标签之间存在函数关系（输入接近某类模板 -> 属于该类），
    因此网络能真正学到分类，用于验证框架的可学习性。

    参数:
        n_samples (int): 样本数。
        n_input (int): 输入维度。
        n_classes (int): 类别数。
        seed (int): 随机种子，保证可复现。

    返回:
        (x, y) 二元组：
        - x: 形状 (n_samples, n_input)，值在 [0,1]（模板 + 噪声后裁剪）；
        - y: 形状 (n_samples,)，标签（长整型，类别索引）。

    使用示例:
        x, y = synthetic_dataset(64, 8, 3)
    """
    g = torch.Generator().manual_seed(seed)
    templates = torch.rand(n_classes, n_input, generator=g)  # 每类一个模板 (n_classes, n_input)
    y = torch.randint(0, n_classes, (n_samples,), generator=g)
    x = templates[y] + 0.2 * torch.randn(n_samples, n_input, generator=g)  # 模板 + 噪声
    x = x.clamp(0, 1)
    return x, y


def load_dvs_gesture(root, T=10):
    """加载 DVS Gesture 数据集（TODO：真实数据下载与预处理尚未实现）。

    TODO（阶段 0 任务 1）: 需要下载 DVS Gesture 并做事件累积/编码。当前阶段 0
    用 synthetic_dataset 演示框架，真实数据在后续接入（或阶段 5 评估实验前接入）。
    """
    raise NotImplementedError(
        "DVS Gesture 数据下载尚未实现。阶段 0 用 synthetic_dataset 演示框架；"
        "真实数据加载留到后续接入。"
    )

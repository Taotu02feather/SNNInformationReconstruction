# -*- coding: utf-8 -*-

import argparse
from utils import setup, train
import logging
import os
import models.layers.surrogate_gradients as gradients
import mlflow
## TESS 的 eligibility trace 中会使用一个二级激活函数，就是这里使用的models.layers.surrogate_gradients

def main():
    parser = argparse.ArgumentParser(description='TESS: A Scalable Temporally and Spatially Local Learning Rule for Spiking Neural Networks')
    # General
    '''
    ### 这一段用于切换各类参数：数据集；模型结构；训练模式（TESS / BPTT）；优化器；学习率策略；STDP 相关超参数；替代梯度函数；运行设备；重复实验次数

    ### 使用多个数据集：CIFAR10、CIFAR100、IBM DVS Gesture、CIFAR10-DVS;使用 VGG-9
        使用 Adam，学习率 0.001;使用不同时间步 T
        使用一组 TESS 相关超参数，如 λpre、λpost、αpre、αpost
        对 αpost 做消融实验

    '''
    parser.add_argument('--arch', type=str, default='cifar_tessvgg_model',
                        help='SNN architecture.')
    ##      架构与运行环境参数
    ##  默认架构名是 cifar_tessvgg_model。这很明显是在指向一个适合 CIFAR 类任务的 TESS-VGG 模型。
    ##  论文实验里多次提到 VGG-9，因此这个参数大概率会在 setup 中映射到 VGG 风格的 SNN 模型构造函数。

    parser.add_argument('--cpu', action='store_true', default=False,
                        help='Disable CUDA training and run training on CPU')

    ##     是否强制使用 CPU
    
    parser.add_argument('--dataset', type=str, default='CIFAR10', choices=['CIFAR10', 'CIFAR100', 'DVSGesture', 'CIFAR10DVS'],
                        help='Choice of the dataset')
    ##      文件直接为论文的四组主实验服务 'CIFAR10', 'CIFAR100', 'DVSGesture', 'CIFAR10DVS'

    parser.add_argument('--save-path', type=str, default='./experiments/default',
                        help='Directory to save the checkpoint and logs of the experiment')
    ##      实验输出目录

    parser.add_argument('--data-path', type=str,
                        help='Path for the datasets folder. The datasets is going to be downloaded if it is not in the location.')
    ##      数据集根目录。如果本地没有，提示信息表明程序会尝试下载。          

    parser.add_argument('--trials', type=int, default=1,
                        help='Number of trial experiments to do (i.e. repetitions with different initializations)')
    ##      训练轮次与复现实验参数
    ##      表示重复做多少次试验，通常是不同随机初始化。
    ##      论文表 II 的消融实验明确写了“五次独立试验”，这个参数就是支撑这类实验复现的入口之一。
    ##      只是 main.py 自己没有显式循环 trials，说明这个参数要么被 train.train() 内部使用，要么只是预留。

    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs to train')
    ##      默认 100。论文实验设置里写的是训练 200 epochs，所以如果要严格复现论文，需要把这个参数改为 200。
    ##      这个差异很重要，说明当前 main.py 默认值不一定等同于论文最终配置。

    parser.add_argument('--optimizer', type=str, choices=['SGD', 'NAG', 'Adam', 'RMSProp', 'RProp'], default='Adam',
                        help='Choice of the optimizer')
    ##      默认 Adam，与论文实验设置一致

    parser.add_argument('--loss', type=str, choices=['MSE', 'BCE', 'CE', 'MSEHW'], default='CE',
                        help='Choice of the loss function')
    ##      默认 CE，也就是分类任务常用交叉熵。

    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Initial learning rate')
    parser.add_argument('--lr-conv', type=float, default=1e-3,
                        help='Initial learning rate')
    ##      两个学习率参数都默认 1e-3。这说明工程可能区分全局学习率和卷积层学习率

    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--val-batch-size', type=int, default=5,
                        help='Batch size for testing')
    ##      论文中的batch size有所不同，要仔细观察

    parser.add_argument('--label-encoding', type=str, default="class", choices=["class", "one-hot"],
                        help='Label encoding by default class. But, one-hot should be use for DFA.')

    parser.add_argument('--activation', type=str, default='LinearSpike', choices=gradients.__dict__["__all__"],
                        help='Name of the secondary activation function (Psi).')
    ##      这个参数直接从 surrogate_gradients 模块里读取候选函数名。说明工程把论文里的 Ψ(u) 设计成可插拔组件。
    ##      论文在实验设置里给了一个具体选择：三角函数
    ##      Ψ(u) = 0.3 · max(1 − |u − vth|, 0)。
    #       而 main.py 默认用 LinearSpike，这很可能就是代码里对应的近似名字，或者只是默认值之一。要判断是否完全等价，需要看 surrogate_gradients 里的实现。

    parser.add_argument('--seed', type=int, default=0,
                        help='Seed for reproducibility. Default 0, which is doesnt set any seed')

    parser.add_argument('--pretrained-model', type=str, default=None,
                        help='Path for the pretrained model')
##          --pretrained-model 预训练模型路径。意味着支持从已有 checkpoint 恢复或做微调。

    parser.add_argument('--training-mode', type=str, default='tess', choices=["tess", "bptt"],
                        help='Training mode.')
    ##      本文中最重要的部分，切换TESS和BPTT


    parser.add_argument('--delay-ls', type=int, default=5,
                        help='Number of time steps for which the learning signal is available (T - T_l).')
    ##      这个参数和论文伪代码里的 t_l 明显对应。论文 Algorithm 1 中写到：
##          当 t >= t_l 时，开始计算局部学习信号 m(l)[t]
##          然后计算 ΔW(l)[t]
##          最后对 t=t_l...T 的更新求和
##          代码里的 delay-ls 表述为 “学习信号可用的时间步数”，本质上是在定义学习信号开始介入的时刻。名称虽然是 delay，但帮助字符串写的是 (T - T_l)，
##          说明作者的内部实现可能以“最后若干时间步启用学习信号”的角度处理。这个参数是论文时序局部学习机制的一个直接实现入口。

    parser.add_argument('--scheduler', type=int, default=0,
                        help='Learning rate decay time.')
##          支持：ReduceLR和Cosine
##          论文写的是：
##      若验证准确率连续 5 个 epoch 不提升，就将学习率减半
##      这显然对应 ReduceLR 类型的策略。因此 main.py 在调度器层面与论文实验设置是一致的。

    parser.add_argument('--print-freq', type=int, default=200,
                        help='Frequency of printing results.')

    parser.add_argument('--factors-stdp', nargs='+', type=float, default=[0.2, 0.75, -1, 1],
                        help='STDP parameters $[lambda_{post}, lambda_{pre}, alpha_{post}, alpha_{pre}]$.')
##      [lambda_post, lambda_pre, alpha_post, alpha_pre] 四个参数的输入。默认值为[0.2, 0.75, -1, 1]

    parser.add_argument('--pooling', type=str, default='MAX', choices=["MAX", "AVG"],
                        help='Pooling layer.')
##      支持 MAX / AVG，默认 MAX。 说明网络架构里池化层类型可切换。
            

    parser.add_argument('--weight-decay', type=float, default=0,
                        help='Weight decay L2 normalization')
##      --weight-decay L2 正则强度，默认 0。



    parser.add_argument('--lr-scheduler-type', type=str, default='ReduceLR', choices=["ReduceLR", "Cosine"],
                        help='Pooling layer.')
##          支持：ReduceLR和Cosine
##          论文写的是：
##      若验证准确率连续 5 个 epoch 不提升，就将学习率减半
##      这显然对应 ReduceLR 类型的策略。因此 main.py 在调度器层面与论文实验设置是一致的。

    parser.add_argument('--experiment-name', type=str, default='TESS',
                        help='Experiment name for mlflow.')
##      实验名称

    parser.add_argument('--wn', action='store_true',
                        help='Use weight normalization')


    parser.add_argument('--avoid-wn', action='store_true',
                        help='Use weight normalization')
##      这俩都是weight normalization


    args = parser.parse_args()



    # Create a new folder in 'args.save_path' to save the results of the experiment
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # Log configuration
    log_path = args.save_path + "/log.log"
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%d-%b-%y %H:%M:%S', filename=log_path)
    logging.getLogger().setLevel(logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler())
    logging.info(args)
    logging.info('=> Everything will be saved to {}'.format(args.save_path))

    # Initiate the training
    experiment_name = args.experiment_name
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run():
        for key in args.__dict__.keys():
            mlflow.log_param(key, args.__dict__[key])
        device, train_loader, test_loader = setup.setup(args)
        train.train(args, device, train_loader, test_loader)


if __name__ == '__main__':
    main()

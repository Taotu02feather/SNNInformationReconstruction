# ABLATION STUDY ON INCLUDING NON-CAUSAL TERMS

## Training Scripts:

- `script_alpha_post.sh`

- `script_CIFAR10DVS_alpha.sh`

`DVSCIFAR10`部分是重复的，因为修复问题后重新对此部分进行了训练

## Result Logs:

Under the Dir:`\experiments\alpha_post_test`

## 分析文件：

- `parse_alpha_ablation.py`

- `plot_alpha_ablation.m`

只接受严格合法的 ablation 目录名，例如：
- `CIFAR10_VGG_TESS_apo_neg1`

- `CIFAR10_VGG_TESS_apo_zero`

- `CIFAR10_VGG_TESS_apo_pos1`

- `CIFAR100_VGG_TESS_apo_neg1`

- `DVSGesture_TESS_apo_pos1`

## 运行方法：

先运行`parse_alpha_ablation.py`，得到`alpha_post_curves`与`alpha_post_summary`;

然后运行`plot_alpha_ablation.m`，将训练结果进行对比，得到`alpha_post_figures`;

然后运行`compare_with_paper.py`，与原论文对比，得到相对应的数据与总结markdown；

最后运行`plot_compare_with_paper.m`，与原论文对比的内容进行画图。

## 数据汇总

### Alpha ablation: paper vs my results

| dataset    |   alpha_post | paper_result      |   my_result   |   delta |
|:-----------|-------------:|:---------------   |------------:  |--------:|
| CIFAR10DVS |           -1 | 75.00 ± 0.69      |       73.9    |   -1.1  |
| CIFAR10DVS |            0 |**75.00 ± 0.65**   |    **74.2**   |   -0.8  |
| CIFAR10DVS |            1 | 74.36 ± 0.87      |       73.5    |   -0.86 |
| DVSGesture |           -1 | 98.56 ± 0.41      |       96.97   |   -1.59 |
| DVSGesture |            0 | 98.33 ± 0.57      |       96.21   |   -2.12 |
| DVSGesture |            1 |**98.56 ± 0.31**   |    **97.35**  |   -1.21 |
| CIFAR10    |           -1 | 89.93 ± 0.31      |       89.67   |   -0.26 |
| CIFAR10    |            0 | 91.99 ± 0.19      |       91.69   |   -0.3  |
| CIFAR10    |            1 |**92.55 ± 0.16**   |    **92.77**  |    0.22 |
| CIFAR100   |           -1 | 62.49 ± 1.05      |       61.7    |   -0.79 |
| CIFAR100   |            0 | 68.19 ± 0.55      |       68.26   |    0.07 |
| CIFAR100   |            1 |**70.00 ± 0.34**   |    **70.14**  |    0.14 |

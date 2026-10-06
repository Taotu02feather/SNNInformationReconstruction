# Comparison of BPTT and TESS across different Image Recognition Tasks

## Training Scripts:

- `script_new.sh`

- `script_CIFAR10DVS_BPTTTESS.sh`

`DVSCIFAR10`部分是重复的，因为修复问题后重新对此部分进行了训练

## Result Logs:

Under the Dir:`\experiments`,

- `CIFAR10_VGG_BPTT_`

- `CIFAR10_VGG_TESS_`

- `CIFAR100_VGG_BPTT_`

- `CIFAR100_VGG_TESS_`

- `CIFAR10DVS_VGG_BPTT_`

- `CIFAR10DVS_VGG_TESS_`

- `DVSGesture_BPTT_`

- `DVSGesture_TESS_`

## 分析文件：

- `parse_bptt_tess_compare.py`

- `plot_bptt_tess_compare.m`


只接受白名单中的 8 个目录名。

例如：

- CIFAR10_VGG_TESS_   Correct

- CIFAR10_VGG_TESS    Wrong

## 运行方法：

先运行`parse_bptt_tess_compare.py`，得到`bptt_tess_curves`与`bptt_tess_summary`;

然后运行`plot_bptt_tess_compare.m`，将训练结果进行对比，得到`bptt_tess_figures`;

然后运行`compare_with_paper.py`，与原论文对比，得到相对应的数据与总结markdown；

最后运行`plot_compare_with_paper.m`，与原论文对比的内容进行画图。

## 数据汇总

### BPTT / S-TLLR / TESS: paper vs my results

| dataset    | method   | paper_result   | my_result   | delta   |
|:-----------|:---------|:---------------|:------------|:--------|
| CIFAR10DVS | BPTT     | 76.40 ± 0.66   | 76.40       | +0.00   |
| CIFAR10DVS | S-TLLR   | 75.14 ± 1.37   | N/A         | N/A     |
| CIFAR10DVS | TESS     | 75.00 ± 0.65   | 75.60       | +0.60   |
| DVSGesture | BPTT     | 97.95 ± 0.68   | 97.35       | -0.60   |
| DVSGesture | S-TLLR   | 98.48 ± 0.37   | N/A         | N/A     |
| DVSGesture | TESS     | 98.56 ± 0.31   | 91.67       | -6.89   |
| CIFAR10    | BPTT     | 92.55 ± 0.06   | 92.70       | +0.15   |
| CIFAR10    | S-TLLR   | 91.88 ± 0.28   | N/A         | N/A     |
| CIFAR10    | TESS     | 92.55 ± 0.16   | 92.48       | -0.07   |
| CIFAR100   | BPTT     | 69.28 ± 0.37   | 69.05       | -0.23   |
| CIFAR100   | S-TLLR   | 68.00 ± 0.71   | N/A         | N/A     |
| CIFAR100   | TESS     | 70.00 ± 0.34   | 69.10       | -0.90   |

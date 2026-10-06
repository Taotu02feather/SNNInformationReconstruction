# BPTT / S-TLLR / TESS: paper vs my results

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

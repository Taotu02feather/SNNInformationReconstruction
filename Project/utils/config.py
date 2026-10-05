"""全局超参配置：膜电位衰减因子、发放阈值等。

这些常量被 snn/lif.py、network/mlp_snn.py 等文件导入作为默认参数。
"""

# LIF 神经元参数
LAMBDA = 0.5   # 膜电位衰减因子（leak）λ，取值 (0, 1)
V_TH = 1.0     # 发放阈值 V_th

# 示例时间步数
T = 3          # sanity/forward_check.py 手算例子使用的时间步数

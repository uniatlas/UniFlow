
# ProxylessNas 概述 #
不使用代理，直接在整个网络上进行搜索，但是又能实现在大数据集直接搜索，这里主要在显存占用这块做了改进
# ProxylessNas 特点 #
- 打破了Block 堆叠的Network的构成方式
- 能直接在大数据集上进行搜索（ImageNet）
- 提出了对于NAS做剪枝的方案，展示了NAS与模型压缩之间的相近的关系，同时通过Path剪枝，本文的方法节约了大约一个量级的计算量
- 提出了一个基于Latency Regularization Loss的梯度下降方法以解决硬件目标问题
# 整体搜索流程 #
# 超参网络结构定义 #

# architectue 和 weight 更新迭代方法 #

### [ProxylessNAS](<https://openreview.net/forum?id=HylVB3AqYm> )

Review can be find in [review](<https://openreview.net/forum?id=HylVB3AqYm> )

Code can be find in [ProxylessNAS](https://github.com/mit-han-lab/ProxylessNAS/blob/master/training/main.py)

#### 1. 论文解析

##### 1. 解决什么问题

- 解决ENAS 或者 NAS 的计算消耗资源过多的问题（ProxylessNAS 是在大数据集上 ImageNet 而不是 CIFAR 这种小数据集）；
- 相对于 ProxyNas 例如 MnasNet 纯代理的 Nas (先训练后迁移,EfficientNet0 类似 MnasNet 搜索出来的 Model)
- 类似于 Darts, 不需要控制控制选择模型
- Differentiable NAS (可微分 NAS )，连续的 NA，GPU的内存使用过大问题，使用代理小任务的话，比如小数据集或者少 block 的话可能不是最优的，然后还需要迁移模型
- 将模型的 latency 作为优化目标之一，在目标数据集而非代理数据集（这里是否对我们具有操作性，获取设备的 latency 值）
- 设备多样性， CPU、GPU、Mobile
- One-Shot 中 drop path 的概率是固定的，而非学习的，并且需要训练评估
- Darts 需要代理任务评估， real-valued architecture parameter for each path and jointly train weight parameters
  and architecture parameters via standard gradient descent.

##### 2. 思想

剪枝（剪枝的作用本来类似网络结构搜索，通过剪枝发现一个有效的网络结构后，剪枝权重没有那么重要），训练一个 over-parameterized network, 通过结构化参数() -》得到需要训练的一个网络，为了处理不可微分的硬件目标如 latency ，使用 REINFORCE_based 算法作为处理硬件度量的策略

##### 3.Trick

1. 如何解决所有 Path 加载到 GPU造成的显存爆炸问题？ over-parameterized network
2. 如何训练 mask? 通过一个 基于 BinayrConnect 梯度方法训练参数
3. 对于不可微硬件指标如何处理？将指标转换成连续的，类似优化 Loss 一样优化(梯度)

##### 4. 概念

1. over-parameterized network： 包含所有的候选路径，使用一个 mask 层(二值 0或1，通过一个 基于 BinayrConnect 梯度方法训练参数) 强制只有一条运行激活，训练量级满足 GPU  显存的要求

   $\mathcal{N}\left(e, \cdots, e_{n}\right)$ 每一个表示一个 DAG, $\mathcal{O}=\left\{o_{i}\right\}$ 表示候选的 op, $\mathcal{N}\left(e=m_{\mathcal{O}}^{1}, \cdots, e_{n}=m_{\mathcal{O}}^{n}\right)$ 表示固定 op,有 N 条并行的路径的图，和 Darts 类似的 mixed Op，故：

   $m_{\mathcal{O}}^{\text { One-Shot }}(x)=\sum_{i=1}^{N} o_{i}(x), \qquad m_{\mathcal{O}}^{\text { DARTS }}(x)=\sum_{i=1}^{N} p_{i} o_{i}(x)=\sum_{i=1}^{N} \frac{\exp \left(\alpha_{i}\right)}{\sum_{j} \exp \left(\alpha_{j}\right)} o_{i}(x)$ 计算的资源和 N 相关

   $m_{\mathcal{O}}^{\mathrm{Binary}}(x)=\sum_{i=1}^{N} g_{i} o_{i}(x)=\left\{\begin{array}{cc}{o_{1}(x)} & {\text { with probability } p_{1}} \\ {\cdots} & {} \\ {o_{N}(x)} & {\text { with probability } p_{N}}\end{array}\right.$ 和 N 无关

   上面可以看出：

   1. One-shot  sum 所有的候选，肯定会带来显存的问题，并且显存随着候选的增大而增大
   2. ProxylessNas 和 Darts 类似的是依概率的，Darts 的依概率是通过 Softmax 软化的，也是随着N 增大而增大，而ProxylessNAS 通过 mask 层只留存一条 path，故显存不随着 N 的增大而增大

2. How to trian?

   1. 确定一个完整的候选网络（但是不激活也没有参数），然后随机采样一个 mask [0,0..1,0,0] ,然后选出来的单路径模型通过标准的梯度下降训练
2. 然后通过验证集上，固定之前训练的模型参数，更新框架参数，比较 prune 掉路径选择最佳路径
   
3. Mask 层不连续如何训练？

   Based the Gradient BinaryConnect algorithm

   $\frac{\partial L}{\partial \alpha_{i}}=\sum_{j=1}^{N} \frac{\partial L}{\partial p_{j}} \frac{\partial p_{j}}{\partial \alpha_{i}} \approx \sum_{j=1}^{N} \frac{\partial L}{\partial g_{j}} \frac{\partial p_{j}}{\partial \alpha_{i}}=\sum_{j=1}^{N} \frac{\partial L}{\partial g_{j}} \frac{\partial\left(\frac{\exp \left(\alpha_{j}\right)}{\sum_{k} \exp \left(\alpha_{k}\right)}\right)}{\partial \alpha_{i}}=\sum_{j=1}^{N} \frac{\partial L}{\partial g_{j}} p_{j}\left(\delta_{i j}-p_{i}\right)$

   这里概率 p_j 使用 g_j 模型的，即用 mask 模拟概率，softmax 求导后做了一次替换，但是这样框架更新计算还是随着 N 增大而增大

4. 如何解决框架更新随着 N 增大而增大的问题？

   有一个先验条件：The intuition is that if a path is the best choice at a particular position, it should be the better choice when solely compared to any other path. 其实就是说我用了多项式分布，虽然分布改了但是这个结果应该是可迁移性的。

   然后从 N 个中选择两个作为N 进行 mask 层的训练，注意最后复原成 N 个

5. 如何处理不可微分的硬件目标？

   latency 而不是 FLOPS

   1. 类似 Darts 软化候选 Op，软化每一个候选Op 的 latency

      $\begin{aligned} \mathbb{E}[\text { Latency }]=& \alpha \times F\left(\operatorname{conv}_{-} 3 \times 3\right)+\\ & \beta \times F\left(\operatorname{conv}_{-} 5 \mathrm{x} 5\right)+\end{aligned}...$

      $\mathbb{E}[\text { latency }]=\sum_{i} \mathbb{E}\left[\text { latency }_{i}\right]$  

      $\text {Loss}=\operatorname{Loss}_{C E}+\lambda_{1}\|w\|_{2}^{2}+\lambda_{2} \mathbb{E}[\text { latency }]$ 

      也就是在 loss 中体现 latency

6. 使用 REINFORCE -base 算法 训练 BinaryConnect weight, 和 Gradient 方法的区别？

   $J(\alpha)=\mathbb{E}_{g \sim \alpha}\left[R\left(\mathcal{N}_{g}\right)\right]=\sum_{j} p_{i} R\left(\mathcal{N}\left(e=o_{i}\right)\right)$

   $\nabla_{\alpha} J(\alpha)=\sum_{i} R\left(\mathcal{N}\left(e=o_{i}\right)\right) \nabla_{\alpha} p_{i}=\sum_{i} R\left(\mathcal{N}\left(e=o_{i}\right)\right) p_{i} \nabla_{\alpha} \log \left(p_{i}\right)$

   $=\mathbb{E}_{g \sim \alpha}\left[R\left(\mathcal{N}_{g}\right) \nabla_{\alpha} \log (p(g))\right] \approx \frac{1}{M} \sum_{i=1}^{M} R\left(\mathcal{N}_{g^{i}}\right) \nabla_{\alpha} \log \left(p\left(g^{i}\right)\right)$

   需要确定 Reward 函数

    

##### 5. 实验

1. Cifar
   1. 使用 tree-structured architecture space that is introduced by Cai et al. (2018b) with PyramidNet (Han et al., 2017) as the backbone， 3层 两个叶节点
   2. 提出 B 和 F 两个超参数调节网络的 宽度和深度，最后输出通道数也是超参
   3. c/o 表示 use of Cutout (DeVries & Taylor, 2017)
   4. Proxyless-G 是用梯度下降训练 mask 选择；Proxyless-R是用增强学习训练 mask 选择
   5. 正常训练 200-300 epoch, 使用 DropPath 后600 epoch
2. Imagenet
   1. 在多个设备测试， GPU v100 batch 8; cpu 2 Xeon(R) 2.4GHz CPU E5-2640 v4 batch 1; 手机使用的是 Google Pixel batch 1
   2. Proxyless-G + LL 代表使用 latency loss
   3. Proxyless-R  中使用 $A C C(m) \times[L A T(m) / T]^{w}$ 准确率和延时作为目标
   4. 使用 MobileNet-V2 作为 backbone
   5. 调整不同的 MBConv 的 kernel 大小{3,5,7} depth 可选 {3,6}
   6. 如果没有 LL Prooxyless-R 将不起作用
   7. GPU 搜的模型是短宽的

#### 2. 代码解析

（Not Complete open source yet）

#### 3. 借鉴

1. Mask 层的应用
2. Mask 层可微处理
3. latency 引入可能带来不可复现性，这也是复现迟迟无法出来的原因，因为需要考虑到硬件、网络等很多的因素
4. 直接在目标数据集而非代理数据集
5. 大量的使用之前 Mnas 使用的 MBConv,是否具有借鉴意义；MBConv larger kernel is good than small kernel,MBConv 在 scaler 到大模型依旧是有效的

#### 4. 引用

- [one-shot NAS](https://zhuanlan.zhihu.com/p/73539339)
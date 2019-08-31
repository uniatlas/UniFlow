* [MDENAS](#mdenas)
   * [1. 论文解析](#1-论文解析)
      * [1. 思想](#1-思想)
      * [2. 框架搜索空间](#2-框架搜索空间)
      * [3. 方法](#3-方法)
      * [4. 实验](#4-实验)
      * [5. 先验的证明](#5-先验的证明)
   * [2. 代码分析](#2-代码分析)
   * [引用](#引用)

Created by [gh-md-toc](https://github.com/ekalinin/github-markdown-toc)

### MDENAS

全称： Multinomial Distribution Learning for Effective Neural Architecture Search 多项式分布学习用于有效的神经结构搜索

[论文](https://arxiv.org/abs/1905.07529) [代码](https://github.com/tanglang96/MDENAS/tree/ee6c8fe3a5ff746775683775a1afcdb8d4229137)

#### 1. 论文解析

##### 1. 思想

1. 之前的 nas 都是在标准的训练验证数据集上进行估计
2. 不一定需要神经网络训练到收敛才算，通过其他方法评估也可以
3. 有一个先验信息：在训练过程中，当一个网络结构 A 的精度比网络结构 B 要好，那么当收敛的时候，网络结构 A 的表现也优于网络结构 B.
4. 每两个节点连接的概率相同，可以被看做是多项式分布
5. 选择 op 或者说路径是使用分布式采样，然后进行分布式学习



##### 2. 框架搜索空间

1. Node 
   1. 三种类型：输入(2) 中间(4) 输出(1)；每一个 Cell 将上一层的输出 Node作为自己的输入 Node，中间 Node 根据 $o^{(i, j)}$ 获得后 concat
   2. 候选： 3x3 max-pooling 、zero、3x3 average pooling、skip connection(identity)、3x3 dilated convolution with rate 2、5x5 dilated convolution with rate 2、5x5 depth-wise seperatable convolution 8个
   3. 对候选 op 进行 element-wise add 操作
   4. 两个节点连接概率由多项式分布决定 $P\left(\text { node }_{1}, \text { node }_{2}\right)$
2. Cell
   1. 当特征空间减半也就是reduce cell 的时候倾向卷积核的个数翻倍
   2. 后选生成后如何缩小 search space 
3. Network
   1. 根据先验信息 Performance Ranking Hypothesis， 在一个小的 stack（6 layers） 上训练，在验证集上用 20 layers 进行验证

##### 3. 方法

1. Sample (两个 node 间)，初始概率为 1/M ,复用 ProxylessNas 中的 mask 层,最后选出一个 路径
   $$
   g=\left\{\begin{array}{l}{\underbrace{[1,0, \ldots, 0]}_{M} \text { with probability } p_{1}} \\ {\underbrace{[0,0, \ldots, 1]}_{M} \text { with probability } p_{M}}\end{array}\right.
   $$

   $$
   o^{(i, j)}=o^{(i, j)} * g=\left\{\begin{array}{l}{o_{1} \text { with probability } p_{1}} \\ {\cdots} \\ {o_{M} \text { with probability } p_{M}}\end{array}\right.
   $$

2. Multinomial Distribution Learning 多项分布学习
   $$
   \Delta \mathcal{H}^{e}=\left[\begin{array}{c}{\left(\overrightarrow{1} \times \mathcal{H}_{1}^{e}-\mathcal{H}^{e}\right)^{T}} \\ {\cdots} \\ {\left(\overrightarrow{1} \times \mathcal{H}_{M}^{e}-\mathcal{H}^{e}\right)^{T}}\end{array}\right]
   $$

   $$
   \Delta \mathcal{H}^{a}=\left[\begin{array}{c}{\left(\overrightarrow{1} \times \mathcal{H}_{1}^{a}-\mathcal{H}^{a}\right)^{T}} \\ {\cdots} \\ {\left(\overrightarrow{1} \times \mathcal{H}_{M}^{a}-\mathcal{H}^{a}\right)^{T}}\end{array}\right]
   $$

   e 和 a 分别代表 epoch 和 accuracy, 我们通过以下函数进行多项分布参数更新：
   $$
   \begin{array}{r}{p_{i} \leftarrow p_{i}+\alpha *\left(\sum_{j} \mathbb{1}\left(\Delta \mathcal{H}_{i, j}^{e}<0, \Delta \mathcal{H}_{i, j}^{a}>0\right)-\right.} \\ {\left.\sum_{i} \mathbb{1}\left(\Delta \mathcal{H}_{i, j}^{e}>0, \Delta \mathcal{H}_{i, j}^{a}<0\right)\right)}\end{array}
   $$
   我们可以看到使用更少的 epoch 以及更高的 accuracy 的话，对于 p 是提升的效果

3. 最终模型选择 p 最大的op进行连接，如果一个 node 有多个 Input 的时候，选择 topk 然后进行 element-wise add 操作

   ```python
   train_data, valid_data = xxx
   model = xxx
   for i in epochs:
     sample operator according to mask layers
     train network with 1 epoch
     validate the model
     caculate the differential of epoch and accuracy
     update the probability
   ```

##### 4. 实验

1. Train 的时候使用 6 个 Cell 堆积，第二、第三个 Cell 是 reduce 层，每一个 Cell 中有4个 node，初始 channel 16，训练100epochs 设置 batch_size=512, SGD with momentum 初始 lr 0.025（annealed down to zero following a cosine schedule） momentum 0.9 weight_decay=3e-4,多项分布超参数 lr 0.01 

2. 在 cifar 在框架评估阶段， 使用 20 个 Cell 堆积 batch_size=96,使用 cutout ,droppath 概率 0.3; 在 imagenet 训练 250 epochs 使用 batch_size=512, weight decay of 3×10−5,and an initial SGD learning rate of 0.1 (decayed by a factor of 0.97 in every epoch)

3. 两种，

   1. 一种是在 cifar 搜索，cell 迁移到 imagenet;使用 two
      initial convolution layers of stride 2 before stacking 14 cells
      with scale reduction (reduction cells) at 1, 2, 6 and 10
   2. 一种是使用 mobilenetv2 的 backbone 直接 search;MBConv kernels {3, 5, 7} and expanding ratios {1, 3, 6}, zero 和 identity被删除

   

##### 5. 先验的证明

​	Kendall 系数  $\tau=\frac{P-Q}{P+Q}$ 的意思是，一致的个数 P 不一致 Q，值域为 [1,-1]；概率 $p_{\tau}=\frac{\tau+1}{2}$  范围 [0,1]，random sample result(loss、epoch、$\tau$ ) from network in test dataset.论文中结果显示 $\tau$ 基本在 0 之上，即 p > 0.5

#### 2. 代码分析

搜索代码未放出，展缓

#### 引用

- https://mp.weixin.qq.com/s/nN7tgkvvkv8B8N-z4yAgcQ

- [distribution learning](https://en.wikipedia.org/wiki/Distribution_learning_theory) 

- [kendall 相关系数](https://en.wikipedia.org/wiki/Kendall_rank_correlation_coefficient) [kendall 相关系数详解](https://www.cnblogs.com/sddai/p/10323561.html)

  

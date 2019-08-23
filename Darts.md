### Darts

* [Darts](#darts)
   * [1. <a href="https://arxiv.org/pdf/1806.09055.pdf" rel="nofollow">Darts Paper</a> Reader](#1-darts-paper-reader)
      * [1. 先验条件](#1-先验条件)
      * [2. 搜索空间](#2-搜索空间)
      * [3. 近似结构梯度](#3-近似结构梯度)
      * [4. 连续结构编码 α -&gt; 推导离散结构](#4-连续结构编码-α---推导离散结构)
      * [5. 实验](#5-实验)
   * [2. 代码解析](#2-代码解析)
      * [1. Search(搜索 Cell)](#1-search搜索-cell)
      * [2. Augment (Stack Cell)](#2-augment-stack-cell)
   * [3. 衍生](#3-衍生)
   * [引用](#引用)

Created by [gh-md-toc](https://github.com/ekalinin/github-markdown-toc)

相比较 ENAS 中使用增强算法，通过验证集的 reward 进行优化，darts 使用梯度下降优化验证集的 loss，

#### 1. [Darts Paper](<https://arxiv.org/pdf/1806.09055.pdf>) Reader

相对于 NAS 或者 ENAS 这种离散化的黑盒优化问题，Darts 将离散集合松弛到连续空间，通过梯度下降在验证集上验证

##### 1. 先验条件

1. 每个 Cell 固定 N 个node 
2. 对于卷积 N 的前两个 node 是输入节点，最后为输出节点、循环神经网络的输入是当前step 和之前的 step 计算所得
3. 为什么到最后有的 node 间连接，有的不连接？ 依赖于 zero operator 

##### 2. 搜索空间

类似 enas 中的 micro 功能但是不是基于 RNN 选择（也就是没有控制器），首先选择 Cell 后按照规则 Stack, 有额外的 zero operator ，代表两个 node 无关联

continuous relaxation(连续松弛操作)：对 softmax 的选择单个操作做松弛，松弛到全部可能

$\overline{o}^{(i, j)}(x)=\sum_{o \in \mathcal{O}} \frac{\exp \left(\alpha_{o}^{(i, j)}\right)}{\sum_{o^{\prime} \in \mathcal{O}} \exp \left(\alpha_{o^{\prime}}^{(i, j)}\right)} o(x)$   $\mathcal{O}$  是候选操作(卷积、池化、零操作等)，$\alpha$ 是参数向量(操作两个节点的混合权重，是连续的需要学习的)， $\overline{o}^{(i, j)}(x)$ 是依概率选择的固定操作。

$\begin{array}{ll}{\min _{\alpha}} & {\mathcal{L}_{v a l}\left(w^{*}(\alpha), \alpha\right)} \\ {\text { s.t. }} & {w^{*}(\alpha)=\operatorname{argmin}_{w} \mathcal{L}_{t r a i n}(w, \alpha)}\end{array}$ 最小化验证集 loss获取 α 参数，其中最小化训练集 loss 获取 w 参数，这里是共同学习，两者随之变动的计算量很大。这里的α类似于一个超参数但是相对于学习率这些超参数维度更大优化更难。

##### 3. 近似结构梯度

第一步更新：$\begin{aligned} & \nabla_{\alpha} \mathcal{L}_{v a l}\left(w^{*}(\alpha), \alpha\right) \\ \approx & \nabla_{\alpha} \mathcal{L}_{v a l}\left(w-\xi \nabla_{w} \mathcal{L}_{t r a i n}(w, \alpha), \alpha\right) \end{aligned}$ 

这一步是一个 trick: 没有使用 inner 优化（因为计算量很大）（这里如果是局部最优 $\begin{aligned} & \nabla_{\alpha} \mathcal{L}_{v a l}\left(w^{*}(\alpha), \alpha\right) \\ = & \nabla_{\alpha} \mathcal{L}_{v a l}\left(w, \alpha\right) \end{aligned}$，则梯度$\nabla_{w} \mathcal{L}_{t r a i n}(w, \alpha)=0$，通过调节learning rate $\xi$ （一般使用和权重相同的 lr） 使用单步的训练参数，而不是argmin Train loss 得到的收敛的参数. 

由上式可得： $\nabla_{\alpha} \mathcal{L}_{v a l}\left(w^{\prime}, \alpha\right)-\xi \nabla_{\alpha, w}^{2} \mathcal{L}_{t r a i n}(w, \alpha) \nabla_{w^{\prime}} \mathcal{L}_{v a l}\left(w^{\prime}, \alpha\right)$ ，其中 $w^{\prime}=w-\xi \nabla_{w} \mathcal{L}_{\operatorname{train}}(w, \alpha)$ 是一步 forward 权重，第二部分的计算还是很昂贵；当 $\xi$ 为0 时为one-order approximation 一阶近似，大于0 的时候为 二阶近似，$\xi$ （一般确定使用 $\epsilon=0.01 /\left\|\nabla_{w^{\prime}} \mathcal{L}_{v a l}\left(w^{\prime}, \alpha\right)\right\|_{2}$ 

$w^{ \pm}=w \pm \epsilon \nabla_{w^{\prime}} \mathcal{L}_{v a l}\left(w^{\prime}, \alpha\right)$ 得到 $\nabla_{\alpha, w}^{2} \mathcal{L}_{t r a i n}(w, \alpha) \nabla_{w^{\prime}} \mathcal{L}_{v a l}\left(w^{\prime}, \alpha\right) \approx \frac{\nabla_{\alpha} \mathcal{L}_{t r a i n}\left(w^{+}, \alpha\right)-\nabla_{\alpha} \mathcal{L}_{t r a i n}\left(w^{-}, \alpha\right)}{2 \epsilon}$  (finite difference approximation 有限差分近似) 两次向前传播，后两次向后传播得到 α，复杂度由 O(α*w) 到 O(α+w)

```python
确定固定的操作集合(arch)
while not converaged do
	更新α参数根据 Val的 loss 计算梯度并进行更新 #bilevel optimization problem Approximation ：近似双层优化
  更新w参数根据 train 的 loss 计算梯度并更新
通过学习 α 得到最终搜索到的框架
```

##### 4. 连续结构编码 α -> 推导离散结构

$\frac{\exp \left(\alpha_{0}^{(i, j)}\right)}{\sum_{o^{\prime} \in \mathcal{O}} \exp \left(\alpha_{o^{\prime}}^{(i, j)}\right)}$ 选择 top-k , 对于卷积神经网络搜索 k=2、对于循环神经网络搜索 k=1 排除 zero operator 

##### 5. 实验

1. 首先进行 Cell 选择，validate  确定最佳 Cell
2. 用这些 Cell 构建 ARCH ， train 后 validate 最佳 (注意这里的 arc 是固定的)
3. 验证 The best cell 的迁移能力

#### 2. 代码解析

##### 1. Search(搜索 Cell)

搜索出的 genotype 注意是两两一组合（这里是不是有局限性或者更多的可能性），最后 concat 操作

1. 首先初始化 alphas（是一个参数矩阵 tensor ，注意个数是和 node 节点个数相同，这里的 node 数目是不包含输入的两个节点以及最后 concat 的节点）, 合格 alphas 会在之后计算 mixOp 的时候作为加权参数

2. 调用 SearchCNN 类生成 net，这是按照一定的规则做的（默认8层，8/3-8*2/3 层间是执行 reduce 操作，同时 channel * 2 ）,调用 SearchCell 搜索 Cell

3. SearchCell 根据传入的前一个 cell 是否 reduce ，是的话 c_{k-1} 节点将进行 FactorizedReduce 操作，c{k-2}  进行 StdConv，构建 dag 的时候需要考虑到两个输入节点，通过 遍历得到一个 dag  注意这个 dag 是类似于paper 中的全部连接并且全部节点上有 MixedOp 构建的所有的待选 op，构建出整个图

   ```python
   for i in range(self.n_nodes):
               self.dag.append(nn.ModuleList())
               for j in range(2+i): # include 2 input nodes
                   # reduction should be used only for input node
                   stride = 2 if reduction and j < 2 else 1
                   op = ops.MixedOp(C, stride)
                   self.dag[i].append(op)
                   
   for primitive in gt.PRIMITIVES:
               op = OPS[primitive](C, stride, affine=False)
               self._ops.append(op
   ```

4. 对于参数 权重w 和 框架α 分别使用 Momentum 和 Adam 进行优化

5. train 方法中 

   ```python
   # architect 已经加载 model, 这里计算展开 loss 并进行梯度下降
   architect.unrolled_backward(train_image, train_label, val_image, val_label, w_optim)
   
   # unrolled_backward 执行以下步骤
   # 1. 主要是更新 w 对应论文中的 等式4 的条件
   self.virtual_step(train_image, train_label, w_optim)
   # 2. 用 val 计算新的 unroll loss，对应论文 等式3 使用验证集
   loss = self.v_net.loss(val_image, val_label)
   # 计算梯度, 这里体现论文中的一句话， both loss is not only determined by architect α but also weight w， 计算联合下降梯度
   v_alphas = tuple(self.v_net.alphas())
   v_weights = tuple(self.v_net.weights())
   v_grads = torch.autograd.grad(loss, v_alphas + v_weights)
   dalpha = v_grads[:len(v_alphas)]
   dw = v_grads[len(v_alphas):]
   
   # 对应论文的等式8，其中 eps 的求解对应论文中的注释2
   hessian = self.compute_hessian(dw, train_image, train_label)
   
   # 最后更新梯度，对应等式 7 update final gradient = dalpha - xi*hessian
   with torch.no_grad():
     for alpha, da, h in zip(self.net.alphas(), dalpha, hessian):
       alpha.grad = da - xi*h
   ```

6. 得到 child model 后进行梯度下降

7. 进行梯度 clip,优化计算准确率

8. 解析最后生成的 Cell `genotypes.py 中的 parse 方法`

##### 2. Augment (Stack Cell) 

1. AugmentCNN

   ```python
   #注意传入搜索出的 genotype, 默认层数为 20 (注意在 20/3 和 40/3 这两层 channel*2 执行 reduce 操作)， 默认使用 aux
   model = AugmentCNN(input_size, input_channels, config.init_channels, n_classes, config.layers,
                          use_aux, config.genotype)
   
   # 注意最后使用 AdaptiveAvgPool2d
   ```

2. AugmentCell

   ```python
   # 
   cell = AugmentCell(genotype, C_pp, C_p, C_cur, reduction_p, reduction)
   # AugmentCell 做以下事情
   #1. 首先两个 node 作为输入, 注意两个 Node 一个时之前的 Cell 的前 node 以及 前前 node,如果有 reduce 操作会在第一个 Node 做
   if reduction_p:
       self.preproc0 = ops.FactorizedReduce(C_pp, C)
   else:
       self.preproc0 = ops.StdConv(C_pp, C, 1, 1, 0)
   self.preproc1 = ops.StdConv(C_p, C, 1, 1, 0)
           
   # generate dag
   if reduction:
   	  gene = genotype.reduce
       self.concat = genotype.reduce_concat
   else:
       gene = genotype.normal
       self.concat = genotype.normal_concat
   self.dag = gt.to_dag(C, gene, reduction)
   ```

3. 生成 DAG

   ```python
   def to_dag(C_in, gene, reduction):
       """ generate discrete ops from gene """
       dag = nn.ModuleList()
       for edges in gene:
           row = nn.ModuleList()
           for op_name, s_idx in edges:
               # reduction cell & from input nodes => stride = 2
               stride = 2 if reduction and s_idx < 2 else 1
               op = ops.OPS[op_name](C_in, stride, True)
               if not isinstance(op, ops.Identity): # Identity does not use drop path
                   op = nn.Sequential(
                       op,
                       ops.DropPath_()
                   )
               op.s_idx = s_idx
               row.append(op)
           dag.append(row)
   
       return dag
   ```

4. 生成DAG 按照正常的 Model 训练

5. 为什么使用 aux_head?  Auxiliary head in 2/3 place of network to let the gradient flow well

#### 3. 衍生

[网络搜索之DARTS, GDAS, DenseNAS, P-DARTS, PC-DARTS](https://mp.weixin.qq.com/s/H30A9cBFkAu4o7amO8SEwg)

#### 引用

- [NAS 综述](<https://blog.csdn.net/c9Yv2cf9I06K2A9E/article/details/82321884>) 翻译于 [automl  org](<https://www.automl.org/book/>)
- <https://github.com/hibayesian/awesome-automl-papers>
- <https://github.com/markdtw/awesome-architecture-search>
- [网络搜索之DARTS, GDAS, DenseNAS, P-DARTS, PC-DARTS](https://mp.weixin.qq.com/s/H30A9cBFkAu4o7amO8SEwg)

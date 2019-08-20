# PC-DARTS 介绍 #
DARTS 在网络体系结构中仍然存在较大的冗余空间，存在较大的内存和计算开销。因为这限制了 DARTS 不能使用太大的 batch size,因此损失了搜索的速度或者高稳定性。有些论文提出减少搜索空间，这导致了一个近似，可能牺牲了发现的架构的最优性。PC-DARTS（Partially-Connected DARTS） 能够有效的减少显存，因为能够用更大的 batch 进行搜索，并且所花费的时间更少而且拥有更高的精度。PC-DARTS 是 DARTS 的衍生版。其核心的思想是：没有将所有通道都发送到操作选择块中，而是随机抽取其中的一个子集进行操作选择，同时直接绕过其余的通道。假设这个子集上的计算是对所有通道上的计算的近似。除了极大地降低了内存和计算成本外，抽样还带来了另一个好处，即操作搜索是规范化的，不太可能陷入局部最优。然而，PC-DARTS也带来了一个副作用，当不同的通道子集在迭代中采样时，网络连接的选择将变得不稳定。因此，我们引入边缘标准化，通过显式学习一组额外的边缘选择超参数来稳定网络连接搜索。通过在整个训练过程中共享这些超参数，所学习的网络体系结构更不容易跨迭代采样通道，从而更稳定。
得益于部分连接策略，我们能够相应地增加批大小。在实际操作中，我们对每个操作选择随机抽取1/K个通道作为样本，这几乎减少了K倍的内存开销。这使得我们可以在搜索过程中使用K×批大小，这不仅加快了搜索速度K倍，而且稳定了搜索，特别是对于大型数据集。
# 相关工作 #
现有NAS方法大致可以分为三类，即基于进化的方法、基于强化学习的方法和 one-shot 方法。
为了在较短的时间内完成架构搜索，研究人员考虑降低评估每个搜索候选对象的成本。早期的工作包括在搜索和新生成的网络之间共享权重，后来这些方法被概括为一个更优雅的框架，称为 one-shot 架构搜索。在这些方法中，只训练一次覆盖所有候选操作的超参数化网络或超网络，并从该超网络的采样中得到最终的体系结构。
![](https://i.imgur.com/gG7WhZt.png)
在此图中解释了信息如何传输到 node3,在模型搜索的过程中主要有2组超参数：α 和 β，对于参数 α 我们只采样了输入的 1/K,剩余的直接 contact 到了下一个 stage.为了消除 1/k 采样造成的不确定影响我们加入了 β，最终是通过 α*β 来评估。

# 方法论 #
- **DARTS：**
论文首先回顾了 DARTS：DARTS 将搜索到的网络分解为若干(L)个单元格。每个单元被组织成一个有 N 个节点的有向无环图(DAG)，其中每个节点定义一个网络层。有一个预定义的操作空间，用O表示，其中每个元素 O(·)是在网络层上执行的固定操作(如恒等连接、3×3卷积)。在单元格中，目标是从O中选择一个操作来连接每对节点。设一对节点为(i,j)，其中 DARTS 的核心思想是将从  i传播到 j 的信息表示为 |O| 操作的加权和。这种设计使得整个框架可以区分层权值和超参数 αo i, j,这样就可以以一个端到端的方式执行架构搜索。搜索过程结束后会保留相应的参数。
- **PC-DARTS：**
DARTS 的一个缺点是它的内存不足。为了适应|O|操作，在搜索架构的主要部分中，需要在每个节点(即，每个网络层)，导致|O|×内存使用。为了适应GPU，必须在搜索过程中减少批量大小，这不可避免地会降低搜索速度，而且更小的批量大小可能会降低搜索的稳定性和准确性。
以图一上看，以xi到xj的连接为例。这包括定义一个通道采样掩码si,j，它将1分配给选定的通道，0分配给掩码通道。所选信道被送入|O|运算的混合计算中，而掩蔽信道则绕过这些运算，即，它们被直接复制到输出，

A、部分通道连接：

如上图的上半部分，在所有的通道数K里随机采样 1/K 出来，进行 operation search，然后operation 混合后的结果与剩下的 (K-1)/K 通道数进行 concat，公式表示如下：
![](https://i.imgur.com/UKYPg4X.png)

B、边缘正规化：

上述的“部分通道连接”操作会带来一些正负作用：

正作用：能减少operations选择时的biases，弱化无参的子操作（Pooling, Skip-connect）的作用。文中3.3节有这么一句话：当proxy dataset非常难时（即ImageNet），往往一开始都会累积很大权重在weight-free operation，故制约了其在ImageNet上直接搜索的性能。

所以可以解释为啥GDAS直接在ImageNet搜效果不行，看回用GDAS在CIFAR10搜出来的normal cell，确实很多是Skip-connect，恐怕在ImageNet上搜，都是skip-connect。。。

副作用：由于网络架构在不同iterations优化是基于随机采样的channels，故最优的edge连通性将会不稳定。

为了克服这个副作用，提出边缘正规化（见上图的下半部分），即把多个PC后的node输入softmax权值叠加，类attention机制：
![](https://i.imgur.com/q3xRZtM.png)

由于 edge 超参 [公式] 在训练阶段是共享的，故学习到的网络更少依赖于不同iterations间的采样到的channels，使得网络搜索过程更稳定。当网络搜索完毕，node间的operation选择由operation-level和edge-level的参数相乘后共同决定。

# 代码解释 #
## genotype结构解析 ##
![](https://i.imgur.com/CbDLasZ.png)
在代码中我们看到 genotype 这个结构出现了很多次，那么这个结构代表的是什么意思呢？取了 genotype 里的一个 normal cell 的定义及其对应的 cell 结构图首先说明下，这个定义的解释。DARTS 搜索的也就是这个定义。normal 定义里(‘sep_conv_3x3’, 1)的 0，1，2，3，4，5 对应到图中的红色字体标注的。从 normal 文字定义两个元组一组，映射到图中一个蓝色方框的节点(这个是作者搜索出来的结构，结构不一样，对应关系不一定是这样的)。
sep_conv_xxxx表示操作，0/1表示输入来源

(‘sep_conv_3x3’, 1), (‘sep_conv_3x3’, 0) —-> 节点0

(‘sep_conv_3x3’, 0), (‘sep_conv_3x3’, 1) —-> 节点1

(‘sep_conv_3x3’, 1), (‘skip_connect’, 0) —-> 节点2

(‘skip_connect’, 0), (‘dil_conv_3x3’, 2) —-> 节点3

normal_concat=[2, 3, 4, 5] —-> cell输出c_{k}
## 搜索过程中的几个关键点 ##
首先明确，PC-DARTS 与 DARTS 一样，搜索实际只搜 cell 内结构，整个模型的网络结构是预定好的，比如多少层，网络宽度，cell 内几个节点等，输入固定是2个节点；
在构建搜索的网络结构时，有几个特别的地方：
1.预构建 cell 时，采用的一个 MixedOp：包含了两个节点所有可能的连接(genotype 中的 PRIMITIVES)；最后 cell 是由 concat 操作组成。
2.初始化了一个 alphas 矩阵，网络做 forward 时，参数传入，在 cell 里使用，搜索过程中所有可能连接都在时，计算mixedOp的输出，采用加权的形式。
# 几个函数解析 #
- ## MixedOp ##

```
lass MixedOp(nn.Module):#MixedOp 函数用于把两节点间的 PRIMITIVES 里定义的所有操作都连接上

  def __init__(self, C, stride):#C 是输入 channel
    super(MixedOp, self).__init__()
    self._ops = nn.ModuleList()
    self.mp = nn.MaxPool2d(2,2)

    for primitive in PRIMITIVES:#PRIMITIVES 指2节点之间的操作
      op = OPS[primitive](C //4, stride, False)
      if 'pool' in primitive:
        op = nn.Sequential(op, nn.BatchNorm2d(C //4, affine=False))
      self._ops.append(op)


  def forward(self, x, weights):
    #channel proportion k=4  
    dim_2 = x.shape[1]
    xtemp = x[ : , :  dim_2//4, :, :]
    xtemp2 = x[ : ,  dim_2//4:, :, :]
    temp1 = sum(w * op(xtemp) for w, op in zip(weights, self._ops))#将所有的输出利用 weight 进行加权
    #reduction cell needs pooling before concat
    if temp1.shape[2] == x.shape[2]:
      ans = torch.cat([temp1,xtemp2],dim=1)
    else:
      ans = torch.cat([temp1,self.mp(xtemp2)], dim=1)
    ans = channel_shuffle(ans,4)
    #ans = torch.cat([ans[ : ,  dim_2//4:, :, :],ans[ : , :  dim_2//4, :, :]],dim=1)
    #except channe shuffle, channel shift also works
    return ans

```
## cell 构建 ##
```
class Cell(nn.Module):

  def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev):
    super(Cell, self).__init__()
    self.reduction = reduction

    if reduction_prev:
      self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
    else:
      self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
    self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
    self._steps = steps#step 等于 4 是固定的
    self._multiplier = multiplier

    self._ops = nn.ModuleList()
    self._bns = nn.ModuleList()
    for i in range(self._steps):#定义 cell  N=6 扣去 2个输入，还剩 4个节点
      for j in range(2+i):#对于节点0会有2个 MixedOp,对于节点1会有3个 MixedOp 依此类推，self._ops 总共会有 14 个 MixedOp
        stride = 2 if reduction and j < 2 else 1
        op = MixedOp(C, stride)
        self._ops.append(op)
  def forward(self, s0, s1, weights,weights2):#构建 cell s0 与 s1 表示2个输入节点
    s0 = self.preprocess0(s0)
    s1 = self.preprocess1(s1)

    states = [s0, s1]#weight[]长度是14，2+3+4+5，所以可以理解 offset 的作用了
    offset = 0
    for i in range(self._steps):
      s = sum(weights2[offset+j]*self._ops[offset+j](h, weights[offset+j]) for j, h in enumerate(states))
      offset += len(states)
      states.append(s)

    return torch.cat(states[-self._multiplier:], dim=1)
```
##  architect函数 ##
architect 函数是代码中对应与论文中的计算公式的重要地方，主要用于对 arch_parameters() 参数的更新，即[
self.alphas_normal,self.alphas_reduce,self.betas_normal,self.betas_reduce,]几个参数。

```
class Architect(object):

  def __init__(self, model, args):
    self.network_momentum = args.momentum
    self.network_weight_decay = args.weight_decay
    self.model = model
    self.optimizer = torch.optim.Adam(self.model.arch_parameters(),
        lr=args.arch_learning_rate, betas=(0.5, 0.999), weight_decay=args.arch_weight_decay)

....
  def step(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer, unrolled):#用于参数的更新
    self.optimizer.zero_grad()
    if unrolled:
        self._backward_step_unrolled(input_train, target_train, input_valid, target_valid, eta, network_optimizer)
    else:
        self._backward_step(input_valid, target_valid)
    self.optimizer.step()

```
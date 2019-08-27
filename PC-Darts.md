### PC-Darts

全称： Partial Channel Connections for Memory-Efficient Differentiable Architecture Search 部分通道连接构建更内存更有效的可微框架搜索

[论文](https://arxiv.org/pdf/1907.05737.pdf) [代码](https://github.com/yuhuixu1993/PC-DARTS)

#### 1. 论文解析

##### 1. 解决什么问题？(相对于 Darts, 和 Darts 相同的不提)

1. 解决 Darts 在搜索阶段显存占用过大问题？（我训练中是存在这个问题的，训练最小的 cifar10 batch_size 只能设置64 在 Titan V 上面），所以拓展到大数据集上基本是不可操作的

##### 2. 思想

​			为解决 Darts 在搜索过程中遇到的内存占用过大问题，提出在 channel 上采样部分（有点类似于 group conv 的思想） -》 会带来对不同 channel 采样期望不一致问题 -》edge normalization 方法，这里会引入一个超参数

##### 3. trick

1. channel sampling 带来两个好处，一个是有一定正则的效果，另一个是不会陷入局部最优
2. edge 影响（不同采样连接的不稳定性），使用 edge normalization 解决
3. 当采样为 channel 的 1/k 的时候，batch_size 相对于 darts 可以是 K倍（但是这样对性能是有影响的）

##### 4. 方法

1. 先验

   1. 假设 channel 上的采样是整个 channel 期望的平均

2. Partial Channel Connections

   ![./images/image-20190822193640833.png)

   使用一个 $\mathbf{S}_{i, j}$ mask 层进行采样

   $f_{i, j}^{\mathrm{PC}}\left(\mathbf{x}_{i} ; \mathbf{S}_{i, j}\right)=\sum_{o \in \mathcal{O}} \frac{\exp \left\{\alpha_{i, j}^{o}\right\}}{\sum_{o^{\prime} \in \mathcal{O}} \exp \left\{\alpha_{i, j}^{o^{\prime}}\right\}} \cdot o\left(\mathbf{S}_{i, j} * \mathbf{x}_{i}\right)+\left(1-\mathbf{S}_{i, j}\right) * \mathbf{x}_{i}$

   上面公式的意思就是先对 mask 层筛选出来的 channel 进行类似 Darts 的行为，后面再讲mask 筛选剩下的 channel 和执行 Cell 后的channel 拼接起来，其中 mask 概率是一个超参数K

3. Edge Normalization

   Sample channel 后带来的影响：

   1. 好的： 
      1. 在 op 选择的偏移性更小了（也就是相同的输入经过不同的 Op 候选后差别变小，很好理解，因为最后k-1/k的输入没有经过 Op 候选） -》 带来一个好的影响（权重过少的op （skip-connect、max-pool）和 权重大的 op（conv） 的不平衡性减少）；
      2. 不平衡性带来的问题：在早期，搜索更愿意权重少的op以保持输出的一致性，而权重大的在优化之前输出是很不一致的；因此权重少的op 在前期积累大量的参数导致权重大的op 即使优化了参数也无法超过前期，这种现象在 proxy 数据集上很难解决；也导致了 Darts 在Imagenet 上无法获得好的搜索结果
   2. 不好的
      1. 随着采样 channel 的改变，连接变得不稳定，引入 edge normalization

   $\mathbf{x}_{j}^{\mathrm{PC}}=\sum_{i<j} \frac{\exp \left\{\beta_{i, j}\right\}}{\sum_{i^{\prime}<j} \exp \left\{\beta_{i^{\prime}, j}\right\}} \cdot f_{i, j}\left(\mathbf{x}_{i}\right)$

   也就是边缘如何连接不仅仅取决于 MixOp 同时和超参数（正则系数$\beta_{i, j}$）,边缘正则计算量是可以忽略的

4. 类似的思想

   1. ProxylessNAS 二值化了Op候选的分布，每一次采样两个 path
   2. PARSEC 也提出一种基于采样优化学习概率分布的方法

   

##### 5. 实验

1. 8个候选 op: 3×3 and 5×5 separable convolution, 3×3 and 5×5 dilated separable convolution, 3×3 max-pooling, 3×3 average-pooling, skip-connect (identity), and zero (none)
2. 类似 shufflenet 稀疏通道的技巧
3. 设置 K = 4在 cifar10
4. 借鉴 p-Darts freeze 模型超参只在前 15 epoch 微调，warm-up 对于网络参数，减缓参数化操作
5. 超参设置：见论文，有一定的借鉴意义
6. 在 Imagenet 上有一个很关键的操作就是在最开始使用3个 步数为 2 的卷积将输入 224 压缩到 28
7. ImageNet 上面使用 8 个 Cell(包含两个 reduce)，这里在数据集做了妥协，只选择 10% 的 image 用于训练权重 2.5% 的数据更新超参，一共训练 50 epochs，architecture hyper-parameters are frozen during the first 35 epochs
8. 为了保存更多的信息，采用1/2 的子采样率是 Cifar10 的两倍
9. 训练 search 使用8Cell, search 使用 12个 Cell(包含两个 reduce)

##### 6. 深入研究

1. K 值的选择 1/8 

#### 2. 代码解析 (imagenet 分析) 基于原始的 darts 实现

见 model_search_imagenet.py 文件

##### 1. NetWork

输入的两个节点会将 224 的图片压缩到 28 而且不改变 channel 

```python
# stem0 和 stem1
self.stem0 = nn.Sequential(
      nn.Conv2d(3, C_curr // 2, kernel_size=3, stride=2, padding=1, bias=False),
      nn.BatchNorm2d(C_curr // 2),
      nn.ReLU(inplace=True),
      nn.Conv2d(C_curr // 2, C_curr, 3, stride=2, padding=1, bias=False),
      nn.BatchNorm2d(C_curr),
    )

    self.stem1 = nn.Sequential(
      nn.ReLU(inplace=True),
      nn.Conv2d(C_curr, C_curr, 3, stride=2, padding=1, bias=False),
      nn.BatchNorm2d(C_curr),
    )
```

##### 2. Cell

Cell 选择和 darts 是基本相同的，根据公式可以看出，但是在 MixedOp 上是不同的，因为用到了 channel_shuffle 对于通道进行 sample

1. 通道选择注意在程序中是 2 ,论文中说的是 4 比较好是 cifar 测试的，在 Imagenet 上是2比较好

   ```python
   def forward(self, x, weights):
       
       dim_2 = x.shape[1]
       xtemp = x[ : , :  dim_2//2, :, :]
       xtemp2 = x[ : ,  dim_2//2:, :, :]
       xtemp3 = x[:,dim_2// 4:dim_2// 2, :, :]
       xtemp4 = x[:,dim_2// 2:, :, :]
       
       # 这里做调试，看看 channel 的多少进行 MixOp 操作合适
       temp1 = sum(w.to(xtemp.device) * op(xtemp) for w, op in zip(weights, self._ops))
       # 为了保证从 Cell 的 0，1，2 到第三个节点的维度是一致的
       if temp1.shape[2] == x.shape[2]:
         #ans = torch.cat([temp1,self.bn(self.conv1(xtemp3))],dim=1)
         #ans = torch.cat([ans,xtemp4],dim=1)
         ans = torch.cat([temp1,xtemp2],dim=1)
         #ans = torch.cat([ans,x[:, 2*dim_2// 4: , :, :]],dim=1)
       else:
         #ans = torch.cat([temp1,self.bn(self.conv1(self.mp(xtemp3)))],dim=1)
         #ans = torch.cat([ans,self.mp(xtemp4)],dim=1)
   
         ans = torch.cat([temp1,self.mp(xtemp2)], dim=1)
   
       ans = channel_shuffle(ans,2)
       return ans
   
   # 是为了近似随机采样(先自己分后随机采样，分两步模拟,实际上实现 channel shuffle 和 channel shift)
   def channel_shuffle(x, groups):
       batchsize, num_channels, height, width = x.data.size()
   o
       channels_per_group = num_channels // groups
       
       # reshape
       x = x.view(batchsize, groups, 
           channels_per_group, height, width)
   
       x = torch.transpose(x, 1, 2).contiguous()
   
       # flatten
       x = x.view(batchsize, -1, height, width)
   
       return x
   ```

2. Cell 后获取整个候选的 Cell 组合，Cell 中有mixed op，现在需要根据 Edge normalization 

   ```python
   # Class Network(Module)中 的 forward
   def forward(self, input):
       s0 = self.stem0(input)
       s1 = self.stem1(s0)
       for i, cell in enumerate(self.cells):
         if cell.reduction:
           # 论文中的α
           weights = F.softmax(self.alphas_reduce, dim=-1)
           n = 3
           start = 2
           # 论文中的β，注意前两个 Cell 
           weights2 = F.softmax(self.betas_reduce[0:2], dim=-1)
           for i in range(self._steps-1):
             end = start + n
             tw2 = F.softmax(self.betas_reduce[start:end], dim=-1)
             start = end
             n += 1
             weights2 = torch.cat([weights2,tw2],dim=0)
         else:
           weights = F.softmax(self.alphas_normal, dim=-1)
           n = 3
           start = 2
           weights2 = F.softmax(self.betas_normal[0:2], dim=-1)
           for i in range(self._steps-1):
             end = start + n
             tw2 = F.softmax(self.betas_normal[start:end], dim=-1)
             start = end
             n += 1
             weights2 = torch.cat([weights2,tw2],dim=0)
         # 查看 cell 的 forward 
         s0, s1 = s1, cell(s0, s1, weights,weights2)
       out = self.global_pooling(s1)
       logits = self.classifier(out.view(out.size(0),-1))
       return logits
     
   # Cell forward
   def forward(self, s0, s1, weights,weights2):
       s0 = self.preprocess0(s0)
       s1 = self.preprocess1(s1)
   
       states = [s0, s1]
       offset = 0
       for i in range(self._steps):
         s = sum(weights2[offset+j].to(self._ops[offset+j](h, weights[offset+j]).device)*self._ops[offset+j](h, weights[offset+j]) for j, h in enumerate(states))
         #s = channel_shuffle(s,4)
         offset += len(states)
         states.append(s)
   
       return torch.cat(states[-self._multiplier:], dim=1)
   
   ```

   实现$\frac{\exp \left\{\beta_{i, j}\right\}}{\sum_{i^{\prime}<j} \exp \left\{\beta_{i^{\prime}, j}\right\}} \text { with } \frac{\exp \left\{\alpha_{i, j}^{o}\right\}}{\sum_{o^{\prime} \in \mathcal{O}} \exp \left\{\alpha_{i, j}^{o}\right\}}$  的过程

3. 优化 

   注意在 epoch > 15 后面才对于 archetect 参数进行优化

   ```python
   if epoch>=15:
         architect.step(input, target, input_search, target_search, lr, optimizer, unrolled=config.unrolled)
   
   # 梯度下降在 architect.py 中
   ```

4. 生成 dag在 model_search_imagenet.py 中和之前的 darts 是不一样的,最后调用 gentypes.py 生成 dag

   ```python
   def genotype(self):
   
       def _parse(weights,weights2):
         gene = []
         n = 2
         start = 0
         for i in range(self._steps):
           end = start + n
           W = weights[start:end].copy()
           W2 = weights2[start:end].copy()
           for j in range(n):
               W[j,:] = W[j,:]*W2[j]
           edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')))[:2]
           
           #edges = sorted(range(i + 2), key=lambda x: -W2[x])[:2]
           for j in edges:
             k_best = None
             for k in range(len(W[j])):
               if k != PRIMITIVES.index('none'):
                 if k_best is None or W[j][k] > W[j][k_best]:
                   k_best = k
             gene.append((PRIMITIVES[k_best], j))
           start = end
           n += 1
         return gene
       n = 3
       start = 2
       weightsr2 = F.softmax(self.betas_reduce[0:2], dim=-1)
       weightsn2 = F.softmax(self.betas_normal[0:2], dim=-1)
       for i in range(self._steps-1):
         end = start + n
         #print(self.betas_reduce[start:end])
         tw2 = F.softmax(self.betas_reduce[start:end], dim=-1)
         tn2 = F.softmax(self.betas_normal[start:end], dim=-1)
         start = end
         n += 1
         weightsr2 = torch.cat([weightsr2,tw2],dim=0)
         weightsn2 = torch.cat([weightsn2,tn2],dim=0)
       gene_normal = _parse(F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy(),weightsn2.data.cpu().numpy())
       gene_reduce = _parse(F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy(),weightsr2.data.cpu().numpy())
   
       concat = range(2+self._steps-self._multiplier, self._steps+2)
       genotype = Genotype(
         normal=gene_normal, normal_concat=concat,
         reduce=gene_reduce, reduce_concat=concat
       )
       return genotype
   ```

   


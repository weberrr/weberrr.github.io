---
layout:     post
title:      经典CTR模型(3)---Embedding+MLP结构下的浅层改造
subtitle:   FNN，PNN，NFM，ONN
date:       2020-01-03
author:     weber
header-img: img/post-bg-ios9-web.jpg
catalog: true
tags:
    - recommender systems
    - CTR
---

这篇总结梳理的是具备 Embedding + MLP 这样结构的模型，之所以称为**浅层改造**，主要原因在于这些模型都是在embedding层的一些改变：使用复杂的操作让模型在浅层尽可能包含更多的信息，降低后续MLP的学习负担。

[Product-based neural networks for user response prediction](https://ieeexplore.ieee.org/abstract/document/7837964)，2016，IEEE


[Neural factorization machines for sparse predictive analytics](https://dl.acm.org/doi/abs/10.1145/3077136.3080777)，2017，SIGIR

[Operation-aware Neural Networks for User Response Prediction](https://www.sciencedirect.com/science/article/pii/S0893608019302850)，2019，Elsevier

# 1. FNN

FNN（Factorization Machine supported Neural Network）是2016年提出的方法。使用 FM 预训练的 embedding 作为 MLP 的输入。本质上也是二阶段模型，与 GBDT+LR 一脉相承。

![FNN](https://tva1.sinaimg.cn/large/00831rSTgy1gd3mcxgk7ej30jp0fhado.jpg)

FNN本身在结构上并不复杂，如上图所示，就是将FM预训练好的Embedding向量直接喂给下游的DNN模型，让DNN来进行更高阶交叉信息的学习。（如果不了解FM可以看上篇文章，给了FM及其实现）

优点：

1. 离线FM预训练可以引入先验知识，同时加速训练；
2. NN省去了学习feature embedding的步骤，计算开销低；

缺点：

1. 非端到端，不利于online learning；
2. FNN中只考虑了特征交叉，并没有保留低阶特征信息；

# 2. PNN

PNN是2016年提出的一种在NN中引入Product Layer的模型，其本质上和FNN类似，都属于Embedding+MLP结构。作者认为，在DNN中特征Embedding通过简单的concat或者add都不足以学习到特征之间复杂的依赖信息，因此PNN通过引入Product Layer来进行更复杂和充分的特征交叉关系的学习。PNN主要包含了IPNN和OPNN两种结构，分别对应特征之间Inner Product的交叉计算和Outer Product的交叉计算方式。

![image-20200323100313143](https://tva1.sinaimg.cn/large/00831rSTly1gd3mtktui5j311c0r2tcm.jpg)

PNN结构显示通过Embedding Lookup得到每个field的Embedding向量，接着将这些向量输入Product Layer，在Product Layer中包含了两部分，一部分是左边的 z ，就是将特征原始的Embedding向量直接保留；另一部分是右侧的 p ，即对应特征之间的product操作；可以看到PNN相比于FNN一个优势就是保留了原始的低阶embedding特征。

优势：

1. 相较于FNN，保留了低阶embedding信息；
2. 通过product引入了更加复杂的特征交互方式；

不足：

1. 计算时间复杂度相对较高

# 3. NFM

NFM（Neural Factorization Machines）也是将FM与NN结合的结构，相较于FNN，NFM是端到端的模型。与PNN不同的是，将PNN中的Product Layer换成了Bi-interaction Pooling来进行特征交叉的学习。

![image-20200323101510964](https://tva1.sinaimg.cn/large/00831rSTgy1gd3n62qd9vj312k0oojzd.jpg)

实现：

```python
class BiInteractionPooling(nn.Module):
    """
      Input shape
        - A 3D tensor with shape:``(batch_size,field_size,embedding_size)``.

      Output shape
        - 3D tensor with shape: ``(batch_size,1,embedding_size)``.
    """

    def __init__(self):
        super(BiInteractionPooling, self).__init__()

    def forward(self, inputs):
        concated_embeds_value = inputs
        square_of_sum = torch.pow(
            torch.sum(concated_embeds_value, dim=1, keepdim=True), 2)
        sum_of_square = torch.sum(
            concated_embeds_value * concated_embeds_value, dim=1, keepdim=True)
        cross_term = 0.5 * (square_of_sum - sum_of_square)
        return cross_term
```

优点：

1. 使用Bi-interaction pooling 可以具备一定的特征交叉能力；

缺点：

1. 直接进行sum pooling会损失一定的信息，可以考虑attention；

# 4. ONN

ONN（Operation-aware Neural Network）也称 NFFM，其实就是 FFM + MLP。ONN沿袭了Embedding+MLP结构。在Embedding层采用Operation-aware Embedding，可以看到对于一个feature，会得到多个embedding结果；在图中以红色虚线为分割，第一列的embedding是feature本身的embedding信息，从第二列开始往后是当前特征与第n个特征交叉所使用的embedding。这两部分concat之后接入MLP得到最后的预测结果。

![](https://tva1.sinaimg.cn/large/00831rSTgy1gd3p5f23pzj31g90u0npd.jpg)

优点：

1. 引入Operation-aware，进一步增加了模型的表达能力。
2. 同时包含了特征的一阶信息和高阶交叉信息。

不足：

1. 复杂度相对较高
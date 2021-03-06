---
layout:     post
title:      深度学习问题杂记
subtitle:   deep learning
date:       2020-02-03
author:     weber
header-img: img/post-bg-ios9-web.jpg
catalog: true
tags:
    - deep learning
    - 面经
---

# 1.Batch Normalization

链接：https://zhuanlan.zhihu.com/p/34879333

## 1.1 BN是什么？

深层的DNN容易发生internal covariate shift问题（上层网络需要不停调整来适应输入的变化，学习率低，同时训练过程容易陷入梯度饱和区），batch normailzation通过将网络层的输入进行normalization，将输入的均值和方差固定在一定范围内，减少了ICS。

## 1.2 BN表达式

$$
\begin{aligned}
\mu =& \sum_{i=1}^mZ^{[l](i)}
\\
\sigma^2 = & \frac{1}{m} \sum_{i=1}^m(Z^{[l](i)}-\mu)^2
\\
\tilde Z^{[l]} = & \gamma·\frac{Z^{[l]}-\mu}{\sqrt{\sigma^2+\epsilon}} + \beta
\\
A^{[l]}=&g^{[l]}(\tilde Z^{[l]})
\end{aligned}
$$

## 1.3 BN的优点

1. 保证每层输入稳定，加速模型收敛
2. 参数不那么敏感，使网络更加稳定
3. BN有一定的正则化效果：每个batch的均值方差不同，随机增加噪音

# 2. 过拟合产生的根本原因

链接：https://www.cnblogs.com/eilearn/p/9203186.html

过拟合的原因：数据太少 + 模型太复杂

**根本原因**

1. 样本数据和真实数据之间存在偏差。

2. 数据太少，无法描述数据的真实分布。
   举个例子，投硬币问题，如果投了10次，都是正面，根据这个数据进行学习，是无法描述分布的。

**为什么增大数据有效**

对于原因1，增大数据的话，数据量大，所有样本的真实分布是相同的（都是人脸），而随机误差可以在一定程度上抵消（不同背景）。

对于原因2，根据统计学大数定律，样本越大，越能反应真实规律。

# 3. 解决过拟合的方法

1. 增加数据（加，或者造，比如原始图像翻转平移拉伸）
2. 使用合适的模型：减少特征维度，网络层数，神经元个数
3. dropout
4. batch normalizatoin（每个batch的均值和方差不同，等于随机增加噪音）
5. l1&l2正则化
6. 早停
7. 多模型集成

# 4. L1 & L2正则化

## 4.1 为什么L1更容易获得稀疏解？

链接：https://zhuanlan.zhihu.com/p/50142573

1. 从图的角度解释：

   L2正则相当于用圆形去逼近目标，L1正则相当于用菱形去逼近目标，更容易引起交点在坐标轴上，导致稀疏解；

2. 从导数的角度解释：

   损失函数为：$J_{L1}(w) = L(w)+ \lambda|w|$ ；$J_{L2}(w)=L(w) + \lambda w^2$

   假设L(w)在0处的偏导数为 d0，则：
   $$
   \frac{\partial}{\partial w} J_{L2}(w)|_{w=0} = d_0
   $$

   $$
   \frac{\partial}{\partial w} J_{L1}(w)|_{w=0^-} = d_0 - \lambda
   $$

   $$
   \frac{\partial}{\partial w} J_{L1}(w)|_{w=0^+} = d_0 + \lambda
   $$

   所以，引入L2，导数仍为 d0，无变化；引入L1，在0附近导数有个突变，存在一个极小值点，优化时容易收敛到极小值点上。

3. 从先验概率分布的角度解释：

   L2相当于样本服从高斯分布，L1相当于样本服从拉普拉斯分布。拉普拉斯分布是一个尖尖的分布，拉普拉斯分布的数据会稀疏。

## 4.2 为什么DL不使用L1而使用L2？

链接：https://www.zhihu.com/question/51822759

L1的损失函数并不是处处可导，所以不能使用随机梯度下降法，而是使用坐标轴下降法。如果使用现有的大部分深度学习框架自带的优化器（主要是 SGD 及其变种）训练，获得不了稀疏解。

实践中如果想获得稀疏解，可以先给目标函数加 L1 正则，然后在训练过程中或者训练结束后把绝对值较小的参数抹成零。

# 5. 正负样本比例不平衡如何解决

链接：https://www.zhihu.com/question/56662976/answer/223881284

链接：https://blog.csdn.net/u011195431/article/details/83008357

分类时，由于训练集合中各样本数量不均衡，导致模型训偏在测试集合上的泛化性不好。

解决方法有：

## 5.1 采样

1. 过采样

通过增加少数类样本的数量来实现样本均衡。

初始版：简单复制。缺点：如果样本数量少，容易过拟合。

改进版：SMOTE，通过插值的方法进行过采样。

2. 欠采样

减少分类中多数类样本的样本数量来实现样本均衡。

缺点：会丢失多数类样本中的一些重要信息。

## 5.2 惩罚权重

给正负样本在损失上按照样本比例赋予不同的权重，小样本量类别权重高，大样本量类别权重低。

## 5.3 集成方法

集成方法指的是在每次生成训练集时使用所有分类中的小样本量，同时从分类中的大样本量中随机抽取数据来与小样本量合并构成训练集，这样反复多次会得到很多训练集和训练模型。最后在应用时，使用组合方法（例如投票、加权投票等）产生分类预测结果。

# 6. AUC计算

链接：https://blog.csdn.net/qq_22238533/article/details/78666436

公式：
$$
AUC = \frac{\sum_{i\in positive class}rank_i-\frac{M(1+M)}{2}}{M\times N }
$$

# 7. 评价方法

p，r，f1

# 8. 梯度爆炸和梯度消失

https://blog.csdn.net/qq_25737169/article/details/78847691

# 9. batch 过大或者过小



# x. 1x1的卷积核的作用

1. 跨通道的特征整合
2. 特征通道的升降维
3. 减少卷积核参数（简化模型）
4. 实现与全连接等价的效果

# x. 生成模型和判别模型

## 2.1 生成式模型

利用生成模型是根据山羊的特征首先学习出一个山羊的模型，然后根据绵羊的特征学习出一个绵羊的模型，然后从这只羊中提取特征，放到山羊模型中看概率是多少，在放到绵羊模型中看概率是多少，哪个大就是哪个。   

常见的生成模型：隐马尔科夫模型、朴素贝叶斯模型、高斯混合模型、 LDA、 Restricted Boltzmann Machine 等。

## 2.2 判别式模型

要确定一个羊是山羊还是绵羊，用判别模型的方法是从历史数据中学习到模型，然后通过提取这只羊的特征来预测出这只羊是山羊的概率，是绵羊的概率。

常见的判别模型有线性回归、对数回归、线性判别分析、支持向量机、 boosting

![2020-2-9_15-14-17](https://tva1.sinaimg.cn/large/00831rSTly1gdjzaoosenj31hj0u0th8.jpg)

# 
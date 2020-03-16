# 介绍
LambdaMART 是一种 Listwise 类型的 LTR 算法，它基于 LambdaRank 算法和 MART (Multiple Additive Regression Tree) 算法，将搜索引擎结果排序问题转化为回归决策树问题。

# MART 算法
MART，即多重增量回归树（Multiple Additive Regression Tree），也称为：
- GBDT(Gradient Boosting Decision Tree)，梯度渐进决策树
- GBRT(Gradient Boosting Regression Tree)，梯度渐进回归树
- TreeNet，决策树网络

从名字，我们可以看出MART的特征：
- 使用决策树预测结果
- 决策树有多个
- 每个树都比之前的树改进一点点，渐进回归，拟合真实结果
#### Boosting （渐进）思想
Boosting思想，即尝试不断迭代弱模型，通过弱模型叠加的方式，渐渐逼近真实情况，起到足够预测真实值的强模型作用。

可以看出，它需要解决两个问题：
1. 如何保证每一次迭代都对解决问题有所帮助？或者说如何确定迭代步骤的拟合方向？
2. 如何将每一次迭代的弱模型有效的叠加起来？

下面，我们通过 AdaBoost （Adaptive boosting，自适应渐进法）来回答这两个问题。
#### AdaBoost
AdaBoost 是 Yoav Freund 和 Robert Schapire 提出的机器学习算法。两人因为该算法获得了 2003 年的哥德尔奖。

AdaBoost 是一种分类算法，其大致过程如下：
1. 制作一个弱分类器 w1 （决策树），去拟合实际情况，如图1的绿线。其中蓝色和红色的圆圈，代表两类样本。
2. 运行 w1， 记录 w1 下被错误分类的样本。赋予这些样本更高的权重（图中表示为圆圈更大），进行第二次拟合，得到弱分类器 w2（图2中的虚线）
3. 依次运行 w1 --- w2 ，记录被错误分类的样本，赋予更高权重，第三次拟合，得弱分类器 w3；
4. 依次运行 w1 --- w2 --- w3 ，如此迭代....
![](https://upload-images.jianshu.io/upload_images/6802002-644c03485af6311a.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
整个过程，用数学符号表达如下：
![](https://upload-images.jianshu.io/upload_images/6802002-86ea299bd7e0b4a4.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/440)
最开始的时候，每个样本点的权重都一致。随着算法不断迭代，被错误分类的样本，权重不断加强，与此同时被正确分类的样本，权重不断减弱。可以想象，越往后，算法越关注那些容易被分错的样本点，从而最终解决整个问题。

回答之前的两个问题：
- AdaBoost 通过调整样本的权值，来确定弱模型下一轮迭代中拟合的方向：提升错误分类的样本的权值，降低正确分类的样本的权值；
- AdaBoost 是一个加法模型，通过将每一轮的弱模型组合叠加起来，得到有效的强模型。
#### MART 的数学原理
MART就是Boost思想下的一种算法框架，目标即寻找强模型 $f(x)$满足：
$$ \hat{f}(x)=arg min_{f(x)}E[L(y,f(x))|x]$$
训练后的MART也是加法模型：
$$ \hat{f}(x)= \hat{f}_M(x)=\sum_{m=1}^M{f_m(x)}$$
![](https://upload-images.jianshu.io/upload_images/6802002-e0190e77fab48b95.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/640)
**如何确定MART中的迭代方向呢？**
假设已经迭代 $m$次，得到$m$棵决策树，将$m$棵决策树的和记为$ \hat{f}_m(x)=\sum_{i=1}^m{f_i(x)}$，则第 $m+1$轮的优化目标：
$$\Delta \hat{f}_{m+1} =   \hat{f}_{m+1} -  \hat{f}_{m} =  f_{m+1}$$
现在我们要求$\Delta  \hat{f}_{m+1}$，引入损失函数
$$L = L((x,y),f)=L(y,f(x)|x)$$
来表示预测函数$f$的预测结果 $y^*=f(x)$ 与真实值 $y$ 之间的差距。那么，我们的目的是在进行第m+1轮预测后，真实值与预测值之间的差距要减小，即
$$\Delta L_{m+1} =   L_{m+1}((x,y), \hat{f}_{m+1})-L_m((x,y), \hat{f}_{m})<0$$
考虑到
$$\Delta L_{m+1} \approx \frac {∂L_m((x,y), \hat{f}_{m})}{∂\hat{f}_{m}} · \hat{f}_{m+1}$$
若取
$$\Delta \hat{f}_{m+1} =  -g_{im}=-\frac {∂L_m((x,y), \hat{f}_{m})}{∂\hat{f}_{m}} $$
则必有$\Delta L_{m+1}<0$，因此，这个$\Delta \hat{f}_{m+1}$就是${f}_{m+1}$拟合的目标，也是损失函数的梯度。

我们引入一个非常小的正数 η，称为「学习度」或者「收缩系数」。如果，我们在每轮迭代中的预测结果前，乘上这么一个学习度；亦即我们将第 m+1 轮拟合的目标，从 $-g_{im}$ 调整为 $-η·g_{im}$。这样一来，我们每次拟合的目标，就变成了损失函数梯度的一部分。
由于 $\Delta L_{m+1}<0$ 仍然成立，经过多次迭代之后，模型依然可以得到一个很好的结果。但是，与引入学习度之前的情况相比较，每次拟合像是「朝着正确的方向只迈出了一小步」。这种 Shrinkage（缩减），主要是为了防止「过拟合」现象。
#### MART小结
MART 是一种 Boosting 思想下的算法框架。它通过加法模型，将每次迭代得到的子模型叠加起来；而**每次迭代拟合的对象都是学习率与损失函数梯度的乘积**。这两点保证了 MART 是一个正确而有效的算法。

MART 中最重要的思想，就是**每次拟合的对象是损失函数的梯度**。值得注意的是，在这里，MART 并不对损失函数的形式做具体规定。实际上，损失函数几乎只需要满足可导这一条件就可以了。这一点非常重要，意味着我们可以把任何合理的可导函数安插在 MART 模型中。LambdaMART 就是用一个 λ 值代替了损失函数的梯度，将 λ 和 MART 结合起来罢了。
# Lambda
Lambda 的设计，最早是由 LambdaRank 从 RankNet 继承而来。因此，我们先要从 RankNet 讲起。
#### RankNet 的创新
Ranking常见的指标都无法求梯度，因此无法直接对评价指标做梯度下降。
RankNet的创新在于，将不适宜用梯度下降求解的Ranking问题，转化为对概率的交叉熵损失函数的优化问题，从而适应于梯度下降法。
RankNet的终极目标是得到一个带参的算分函数：
$$s=f(x;w)$$
由算分函数可以得到文档的得分：
$$s_i=f(x_i;w) ; s_j=f(x_j;w)$$
根据得分，把排序问题转换成比较一个 (i, j) pair的排序概率问题，计算二者的偏序概率：
$$Pij=P(xi⊳xj)=\frac {exp(σ⋅(s_i−s_j))}{1+exp(σ⋅(s_i−s_j))}=\frac {1}{1+exp(−σ⋅(s_i−s_j))}$$
再定义交叉熵为损失函数：
$$L_{ij}=−\hat{P}_{ij}logP_{ij}−(1−\hat{P}_{ij})log(1−P_{ij})=\frac12(1−S_{ij})σ⋅(s_i−s_j)+log\{1+exp(−σ⋅(s_i−s_j)) \} $$
再梯度下降：
$$w_k→w_k−η \frac{∂L}{∂w_k}$$
#### 梯度观察
![](https://upload-images.jianshu.io/upload_images/6802002-e733de526a008adf.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
这里每条横线代表一条文档，其中蓝色的表示相关的文档，灰色的则表示不相关的文档。在某次迭代中，RankNet 将文档的顺序从左边调整到了右边。于是我们可以看到：

- RankNet 的梯度下降表现在结果的整体变化中是逆序对的下降：13 → 11
- RankNet 的梯度下降表现在单条结果的变化中，是结果在列表中的移动趋势（图中黑色箭头）

我们通常更关注前几条文档的排序情况，因此我们会**期待真正的移动趋势如图中红色箭头所示**
那么问题就来了：我们能不能直接定义梯度呢？
#### LambdaRANK
现在的情况是这样：
- RankNet 告诉我们如何绕开 NDCG 等无法求导的评价指标得到一个**可用的梯度**；
- 上一节我们明确了我们**需要怎样的梯度**；
- 梯度（红色箭头）反应的是某条结果排序变化的趋势和强度；
- 结果排序最终由模型得分 s 确定。

ResNet的梯度为：
$$\frac {∂L}{∂w_k}=\sum_{(i,j)∈P}\frac{∂L_{ij}}{∂w_k}=\sum_{(i,j)∈P}\frac{∂L_{ij}}{∂s_i}·\frac{∂s_i}{∂w_k}+\frac{∂L_{ij}}{∂s_j}·\frac{∂s_j}{∂w_k}$$
注意有下面对称性:
$$\frac{∂L_{ij}}{∂s_i}=-\frac{∂L_{ij}}{∂s_j}$$
我们定义符号$\lambda_{ij}$：
$$\lambda_{ij}=\frac{∂L_{ij}}{∂s_i}=-\frac{∂L_{ij}}{∂s_j}=\frac {\sigma}{1+exp(\sigma·(s_i-s_j))}$$
在此基础上，考虑评价指标 Z（比如 NDCG）的变化：
$$\lambda_{ij}=\frac {\sigma}{1+exp(\sigma·(s_i-s_j))}·|\Delta Z_{ij}|$$
对于具体的文档 xi，有:
$$λ_i=\sum _{(i,j)∈P}λ_{ij}−\sum _{(j,i)∈P}λ_{ij}$$
即**每条文档移动的方向和趋势取决于其他所有与之 label 不同的文档。**

现在回过头来，看看我们做了什么？
- 分析了梯度的物理意义；
- 绕开损失函数，直接定义梯度。

 LambdaRank 的损失函数：
$$L_{ij}=log \{ 1+exp(−σ⋅(s_i−s_j)) \}⋅|ΔZ_{ij}|$$
#LambdaMART
现在的情况变成了这样：
- MART 是一个框架，缺一个「梯度」；
- LambdaRank 定义了一个「梯度」。

让他们在一起吧！于是，就有了 LambdaMART。
![](https://upload-images.jianshu.io/upload_images/6802002-c8b457aae2424c55.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
#### 优点
LambdaMART 有很多优点，取一些列举如下：

直接求解排序问题，而不是用分类或者回归的方法；
可以将 NDCG 之类的不可求导的 IR 指标转换为可导的损失函数，具有明确的物理意义；
可以在已有模型的基础上进行 Continue Training；
每次迭代选取 gain 最大的特征进行梯度下降，因此可以学到不同特征的组合情况，并体现出特征的重要程度（特征选择）；
对正例和负例的数量比例不敏感。

# 实验与应用
#### Ranklib的lambdaMART实现
网上较多的关于lambdaMART的实现资料是对ranklib开源库中对lambdaMART实现的源码解析，我觉得讲的最详细的是这篇博客，原理和代码结合讲的很详细：
[https://www.cnblogs.com/bentuwuying/p/6701027.html](https://www.cnblogs.com/bentuwuying/p/6701027.html)
但Ranklib的实现较早，本文中就不再赘述。
#### pyltr的lambdaMART实现
pyltr听名字就知道，是python实现的learning to rank的开源库
github库地址：[https://github.com/jma127/pyltr](https://github.com/jma127/pyltr)
调用方法：
1. 导入pyltr
```
import pyltr
```
2. 导入数据集（[LETOR](http://research.microsoft.com/en-us/um/beijing/projects/letor/) dataset (e.g. [MQ2007](http://research.microsoft.com/en-us/um/beijing/projects/letor/LETOR4.0/Data/MQ2007.rar) )为例）
```
with open('train.txt') as trainfile, \
        open('vali.txt') as valifile, \
        open('test.txt') as evalfile:
    TX, Ty, Tqids, _ = pyltr.data.letor.read_dataset(trainfile)
    VX, Vy, Vqids, _ = pyltr.data.letor.read_dataset(valifile)
    EX, Ey, Eqids, _ = pyltr.data.letor.read_dataset(evalfile)
```
3. 使用用于早期停止和修剪的验证集，为  LambdaMART 模型定型：
```python
metric = pyltr.metrics.NDCG(k=10)

# Only needed if you want to perform validation (early stopping & trimming)
monitor = pyltr.models.monitors.ValidationMonitor(
    VX, Vy, Vqids, metric=metric, stop_after=250)

model = pyltr.models.LambdaMART(
    metric=metric,
    n_estimators=1000,
    learning_rate=0.02,
    max_features=0.5,
    query_subsample=0.5,
    max_leaf_nodes=10,
    min_samples_leaf=64,
    verbose=1,
)

model.fit(TX, Ty, Tqids, monitor=monitor)
```
4. 对测试数据进行评估：
```
Epred = model.predict(EX)
print('Random ranking:', metric.calc_mean_random(Eqids, Ey))
print('Our model:', metric.calc_mean(Eqids, Ey, Epred))
```
# 参考链接
[LambdaMART 不太简短之介绍](https://liam.page/2016/07/10/a-not-so-simple-introduction-to-lambdamart/)

[机器学习算法-L2R进一步了解](https://jiayi797.github.io/2017/09/25/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E7%AE%97%E6%B3%95-L2R%E8%BF%9B%E4%B8%80%E6%AD%A5%E4%BA%86%E8%A7%A3/)
[Learning To Rank之LambdaMART的前世今生](https://blog.csdn.net/huagong_adu/article/details/40710305)

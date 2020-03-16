
# LDA(Latent Dirichlet Allocation)简介
LDA是一种文档主题生成模型，由Blei, David M, Ng Andrew, Y Jordan于2003年提出，用来推测文档的主题分布。它可以将文档集中每篇文档的主题以概率分布的形式给出，从而通过分析一些文档抽取出它们的主题分布后，便可以根据主题分布进行主题聚类或文本分类。

因为LDA的说明要求的数学基础较多，且不属于RecSys的重点，在这里不做详细介绍，感兴趣的话可以去阅读LDA数学八卦的详细介绍。
链接：[LDA数学八卦索引及全文文档](https://zhuanlan.zhihu.com/p/57418059)

我们只需要知道，当你给定一篇文档内容 $w$ 和主题个数 $n$，通过LDA算法，可以帮你得到文档对应到不同主题的概率：
$$w_i = <t_1,t_2,...,t_n>$$
目前LDA算法已经有了封装实现，调用scikit-learn,spark MLlib和gensim库都有LDA主题模型的类库，本文简单示例scikit-learn中LDA主题模型的使用。

```python3
from sklearn.decomposition import LatentDirichletAllocation

lda = LatentDirichletAllocation(n_topics=2)
docres = lda.fit_transform(cntTf)

print(docres)  #文档的主题分布
print(lda.components_) #主题和词的分布
```


# CTR(Collaborative Topic Regression)
论文：[Collaborative Topic Modeling for Recommending Scientific Articles](http://xueshu.baidu.com/usercenter/paper/show?paperid=3b326a04e93f1383cd631cb476acac33&site=xueshu_se&hitarticle=1)，SIGKDD，2007

CTR模型可以概括为：$CTR=PMF+LDA$
之前说过，传统的PMF的做法是假设物品$V_j∼N(0,\sigma^2_v)$，**仅考虑用户和物品的交互信息**（也叫做协同信息），不使用其他额外信息，通过矩阵分解，将物品$V_j$表示为一个隐向量。

但是往往，我们可以获得一些物品的**内容信息**，比如电商推荐情景下，我们可以获得商品的描述；电影推荐场景下，我们可以获得电影的简介等。CTR的核心思想就是充分利用物品的协同信息（history interaction）和内容信息（side information），用这两部分求和，来一起表示物品的隐向量：
$$V_j=\epsilon_j+θ_j$$
其中$θ_j$是通过LDA算法从物品的内容信息中获取的物品的主题向量表示，
$\epsilon_j∼N(0,\sigma^2_v)$，用来模拟协同信息下用户和物品的交互导致的隐向量偏移。
**评分估计值：**
$$E[r_{ij}|u_i,\epsilon_j,\theta_j]=u_i^T(\epsilon_j+θ_j)$$

**损失函数：**
$$Loss_{ctr}=∑_i∑_j\frac{I_{ij}}2(r_{ij}−u^T_iv_j)^2+\frac{λ_u}2∑_i||u_i||^2_2+\frac{λ_v}2∑_j||v_j−θ_j||^2_2-∑_j∑_nln(∑_k\theta_{jk}\beta_{k,w_{jn}})$$
损失函数的最后一项为LDA的损失，是根据文档-主题, 主题-词汇的联合概率计算得到。从损失函数可以看出，相比于PMF，主要在于项$||v_j−θ_j||^2$，即希望物品的向量表示尽可能接近其基础的内容表示。
![LDA部分的损失忽略](https://upload-images.jianshu.io/upload_images/6802002-c6de66d8bdc790f4.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
在实际实验中，使用固定的$\theta_j$表示可以有效的描述内容信息，并且训练时节省计算，所以一般会先提取内容的topic描述，不再将这部分进行迭代。上面是作者原话。


![CTR](https://upload-images.jianshu.io/upload_images/6802002-4bb3705d6455dab4.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
图中深色框表示已知的标签信息，r 为用户-文章评分，w 为文章的内容。
不同于梯度下降，文章作者使用的 **块坐标下降法（block coordinate descent）** 来进行迭代。
$u_i ←(VC_iV^T+\lambda_uI_k)^{-1}VC_iR_i$
$v_j←(UC_jU^T+\lambda_vI_k)^{-1}(UC_jR_j+\lambda_v\theta_j)$
# CTR优点
1. 增加了内容信息，有效缓解冷启动问题。
2. CTR模型能够提供非常好的解释性：利用LDA主题模型学习到的主题来解释用户隐空间。每个用户交互的物品有内容信息（若干个词），内容信息有对应的主题，选出表征该主题的权重值最大的若干个词，作为用户的画像标签，为用户建立画像标签增加了可解释性。

# 代码实现
用citeulike-a数据集实现了CTR
[https://github.com/weberrr/recsys_model](https://github.com/weberrr/recsys_model)


# 参考链接
[蘑菇先生学习记-CTR协同主题回归](http://xtf615.com/2018/08/15/CTR/)



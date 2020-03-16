# 论文
论文： [http://www.eurecom.fr/en/publication/5290/download/data-publi-5290.pdf](http://www.eurecom.fr/en/publication/5290/download/data-publi-5290.pdf)
代码：[https://github.com/D2KLab/entity2rec](https://github.com/D2KLab/entity2rec)

# 会议
2017，recsys

# 摘要
本文提出了entity2rec，一种从知识图谱中学习用户-物品相关性的方法，用于前N项推荐。文章基于node2vec方法无监督学习知识图谱中实体的属性特定向量表示，再通过学习排名算法给出前N项推荐。最后在MovieLens 1M数据集上与四个基线进行了实验对比。

# Node2vec概要
论文:[node2vec: Scalable Feature Learning for Networks](https://www.kdd.org/kdd2016/papers/files/rfp0218-groverA.pdf)

node2vec通过在图上模拟随机游走，生成节点序列，然后将其输入神经语言模型（word2vec），好像它们是文档的“句子”以学习节点的矢量表示。根据这些表示，可以使用矢量相似性度量容易地计算两个节点之间的相关性。
# entity2rec做法
##### 知识图谱构建
知识图谱定义
$$K=(E,R,O)$$
E为实体集合（set of entities），包括两部分。
用户实体：$user：u \in U \subset E$
物品实体：$item：i \in I \subset E $
R为实体间关系三元组集合:$R \subset E \times \Gamma \times E$
O为本体。
O定义了关系集合$\Gamma$,实体类型集合$\Lambda$
O映射了实体与实体类型：$O:e \in E \to \Lambda$
O映射了实体类型与类型拥有的属性：$O:\epsilon \in  \Lambda \to \Gamma_{\epsilon} \subset \Gamma$
$K$中每条边，为三元组$(i,p,j)$

文章在知识图谱中加入了额外的用户与物品间的属性$ p='feedback' $，所以$p \in \Gamma_{\epsilon}^{+} =\Gamma \bigcup 'feedback'$

文章使用DBpedia本体库来表示movieLens中的实体本体，并为Film类型的实体通过网络链接在线添加其属性。
##### 特定属性的用户-物品相关性计算
node2vec是通过在图上的随机游走生成节点序列，将序列作为输入放入语言模型中训练，得出节点的矢量表示。但是在知识图谱中，**节点的不同属性具有不同的语义值**，在判断实体节点间的相关性时，不同属性也应具有不同的权值。
所以，**先考虑一个特定属性的相关节点的矢量表示**。

对属性$p \in \Gamma_{\epsilon}^+$,
定义子图$K_p$为属性$p$连接的实体节点集合：如$(i,p,j)$中的实体 $i$ 和 $j$ 
对每个子图$K_p$，学习映射$x_p:e \in K_p \to R^d$(即把该Kp中的每个实体e都变为一个矢量表示xp(e))
优化目标函数：
$$max_{x_p} \sum_{e \in K_p} (-logZ_e+\sum_{n_i \in N(e)}x_p(n_i)·x_p(e)) $$
其中 $Z_e = \sum_{v \in K_p} exp(x_p(e)·x_p(v))$，是每个节点和节点e的矢量点积，类似于归一化因子（具体原理见node2vec）。
特定属性下的相关性得分计算公式：
$$p_p{(u,i)}=\begin{cases}
        cos(x_p(u),x_p(i)),  & \text{if p = 'feedback'} \\
        \frac{1}{|R_+(u)|} \sum_{i' \in R_+(u)} {cos(x_p(i),x_p(i'))} , & \text{otherwise}
        \end{cases}$$
其中$R_+(u)$表示用户$u$之前反馈过的物品集合。**当p=feedback时，模拟协同过滤。**
##### 全局用户-物品相关性计算
对所有的用户-物品实体对，计算特定属性相关性得分，得集合 $\tilde{p}(u,i)=\{p_p{(u,i)} \}_{p \in \Gamma_{\epsilon}^+}$，用这些分数作为全局用户-物品相关性模型下进行物品推荐的特征值。定义全局用户物品相关性$p(u,i;\theta)=f(\tilde{p}(u,i);\theta)$，参数$\theta$为优化前N项推荐的参数。

**训练数据：$T=\{(u_k,\tilde{i_k},\tilde{y_k})_{k=1}^N\ \} $**
其中，用户集合$U=\lbrace u_1,u_2,...,u_N \rbrace$ ，每个用户$ u_k$连接'feedback'的物品集合$\tilde{i_k}=\lbrace i_{k\ 1},i_{k\ 2},...,i_{k\ n(k)} \rbrace$，并有反馈评分标签$\tilde{y_k}=\lbrace y_{k\ 1},y_{k\ 2},...,y_{k\ n(k)} \rbrace$ 

**排名函数：$p(u,i,\theta)$**
其中，$p(u,i,\theta)$表示每个用户下，相应物品的分数排名。其引起一串整数排列$\pi(u_k,\tilde{i_k},\theta)$，根据物品$\tilde{i_k}$得分对其排序。本文使用 Adarank 和 LambdaMart 作为排名算法。

**损失函数：$C(\theta)=\sum_{k=1}^N(1-M(\pi(u_k,\tilde{i_k},\theta),\tilde{y_k}))$**
其中，$M(\pi(u_k,\tilde{i_k},\theta),\tilde{y_k}))$表示的是$\pi(u_k,\tilde{i_k},\theta)$与真实值$\tilde{y_k}$间的不同。

**优化：$\theta=arg \min_{\theta} C(\theta)$**
学习过程的目标是找到最小化训练数据上的损
失函数C的参数集θ

# entity2rec实现
具体实现的流程如图所示
（1）从movielens中获取数据，将 user(id) - item(id) - score 的原数据 转成 user(id) - feedback - item(DBpedia_link) 的三元组数据，以 user- item 键值对的形式存储在 feedback.edgelist 文件中；
（2）从DBpedia中获取数据，获取相关的其他属性下的 edgelist 数据，如 starring,editting 等相关数据；
（3）计算特定属性（如：feedback）下的用户-物品相关性矩阵：利用node2vec原理训练特定超参数、维度下的用户x物品向量矩阵；
（4）得到特定属性下的用户物品评分；
（5）对评分进行排序；
![工作流](https://upload-images.jianshu.io/upload_images/6802002-c3aafa48001ce193.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

# 总结
1. entity2rec吸取了node2vec的思想，将关系数据转化为图数据进行训练，这点值得学习！
2. 实验代码写的很认真，实验内容丰富，数据集涵盖LibraryThing，LastFM，movieLens，基线实验涵盖MostPop，NMF，SVD，itemKNN，值得复用学习；
3. entity2rec利用 learning to rank 排序算法将结果进行了进一步加强，可以学习这种方法，以后将排序算法普遍应用于推荐算法上。


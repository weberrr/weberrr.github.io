论文：[KGAT: Knowledge Graph Attention Network for Recommendation](http://xueshu.baidu.com/usercenter/paper/show?paperid=181502a0nr670g90yj080mt003320950&site=xueshu_se&hitarticle=1)，KDD 2019，Xiang Wang， Xiangnan He
****
作者在知识图谱（KG）与用户-物品图混合形成的协同知识图（Collaborative knowledge graph，CKG）中，提出了一种称为**知识图注意力网络（Knowledge Graph Attention Network，KGAT）**的新方法，显示建模图中的高阶连通性，通过递归邻居传播学习节点嵌入，并采用注意力机制来区分邻居嵌入的重要性。
 # 1. 相关介绍
如图所示，传统的协同过滤（Collaborative Filtering，CF）方法对$u_1$的推荐只会考虑$i_1$，但其实观察深层关系，用户$u_2, u_3$以及物品$i_3, i_4$都是对$u_1$进行推荐时非常重要的用户和物品。![](https://upload-images.jianshu.io/upload_images/6802002-051deff5c545b70e.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)这种用户-物品图与知识图谱相结合的图结构为**协同知识图（Collaborative knowledge graph，CKG）**。
**目前利用CKG的方法分为两类：**
**1.基于路径（path-based）**：需要设计路径选择算法或手工设计元路径，来处理两个节点间的大量路径。最终提取出含有高阶信息的路径进行训练。
**2.基于正则化（regularization-based）**：设计其他正则项来拟合KG的结构信息，如MKR。通过这样隐式的方式对高阶信息进行编码，因此难以捕获远程连接，并且可解释性差。

基于现有方法的局限性，作者提出**知识图注意力网络（KGAT）**。
1.与基于路径的方法相比，避免了人工设计路径的过程，效果更高；
2.与基于正则化的方法相比，它直接将高阶关系分解为预测模型，因此所有参数都经过学习以优化推荐目标。

# 2. 问题定义
**符号定义**
用户-物品双向图：$G_1=\{ (u,y_{ui},i|u \in U,i \in I) \}$
知识图：$G_2 = \{ (h,r,t)|h,t \in E,r \in R \}$
物品-实体对齐方式：$A = \{(i,e)|i \in I,e \in E \}$
协同知识图CKE：$G = \{ (h,r,t)|h,t \in E',r \in R' \}$
其中$E'=E \cup U$，$R'=R \cup \{ interact \}$
**任务描述**
输入：$G$
输出：$\hat{y}_{ui}$
# 3. KGAT结构
KGAT结构如图所示，分三个部分：
**Embeddings 层**：通过CKG的结构初始化每个节点的embedding；
**Attentive Embedding Propagation 层**：通过递归方式传播节点邻居的embedding以更新其表示，并采用注意力机制学习传播过程中每个邻居的权重；
高阶连通图来训练embedding
**Prediction 层**：聚合embedding表示，输出预测值
![](https://upload-images.jianshu.io/upload_images/6802002-0985baa3bc311489.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
## 3.1 Embedding层
使用 TransR 学习节点和边的向量表示。

![](https://upload-images.jianshu.io/upload_images/6802002-08b940c082afb188.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
## 3.2 Attentive Embedding Propagation层
**注意力嵌入传播**多层做法与第一层相同，只是重复多次，这里以一层为例。
第一层的传播分为三步：**消息传播**，**基于知识的注意力**，**信息聚合**。
### 消息传播
对于一个结点$h$，记以其为头结点的三元组集合为$N_h= \{(h,r,t)|(h,r,t) \in G  \}$，
则产生的来自邻居的消息为：
$$e_{N_h}=\sum _{(h,r,t)\in N_h} \pi(h,r,t)e_t$$其中，$\pi(h,r,t)$为权值。
### 基于知识的注意力
通过关系注意力机制实现权值：
$$\pi(h,r,t)=(W_re_t)^⊤tanh((W_re_h+e_r))$$注意力得分取决于$r$关系下$e_h$与$e_t$之间的距离。
>作者为简单起见，仅在这些表示形式上使用内部乘积，并将对注意力模块的进一步探索作为将来的工作。

最后，使用softmax标准化：
$$\pi(h,r,t)=softmax_{(h',r',t') \in N_h}$$
### 信息聚合
最后使用聚合器来聚合信息的表示：$e_h^{(1)}=f(e_h,e_{N_h})$
类似于KGCN，作者也设计了三种聚合器：
GCN聚合器：$$f_{GCN}=LeakyReLU(W(e_h+e_{N_h}))$$GraphSage聚合器：$$f_{GraphSage}=LeakyReLU(W(e_h||e_{N_h}))$$Bi-Interaction聚合器：
$$f_{Bi-Interaction}=LeakyReLU(W(e_h+e_{N_h}))+LeakyReLU(W(e_h⊙e_{N_h}))$$实验结果上看是第三种聚合器效果最好。
### 高阶传播
$$e_h^{(l)}=f(e_h^{(l-1)},e_{N_h}^{(l-1)})$$
## 3.3 Prediction层
用户表示：$e_u^*=e_u^{(0)}||···||e_u^{(L)}$
物品表示：$e_i^*=e_i^{(0)}||···||e_i^{(L)}$
其中，$||·||$表示拼接
预测结果为：$\hat{y}(u,i)=e_u^⊤e_i^*$
# 损失函数
**损失：**
$L_{KGAT}=L_{KG}+L_{CF}+\lambda ||Θ||^2_2$
其中，KG部分为TransR的正负样本差损失，CF为pairwise的损失，最后为正则项。
**训练：**
交替优化$L_{KG}$，$L_{CF}$

# 4. 实验结果
**数据集：**
Amazon-book：book
Last-FM：music
Yelp2018： business
>使用 KB4Rec 抽取的Freebase连接作为Amazon-book，Last-FM的KG。对于 Yelp2018，从本地业务信息网络（例如，类别，位置和属性）提取商品知识作为 KG 数据。

![](https://upload-images.jianshu.io/upload_images/6802002-f44def3ae6140075.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
**实验结果：**

![](https://upload-images.jianshu.io/upload_images/6802002-50868f684599bcea.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

**参数分析：**
分析了embedding的层数，聚合器，以及使用图嵌入和attention的效果；

![](https://upload-images.jianshu.io/upload_images/6802002-ffe0a29f7e2177c1.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
![](https://upload-images.jianshu.io/upload_images/6802002-eca3fe9630663033.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


**时间效率：**
推荐往往讲究实时性，作者还分析了时间效率：
FM，NFM，CFKG，CKE，GC-MC，KGAT，MCRec 和 RippleNet 的成本分别约为
700s，780s，800s，420s，500s，560s，20 小时和 2 小时。
# 5. 总结
KGAT在结构上和NGCF很相近，算是NGCF的进化版。
KGAT的想法和我很相近，希望结合CF信息和KG信息进行推荐。
但KGAT始终是GNN的壳子，表示完每层向量之后过通过多层的迭代训练得到向量。这个GNN的思想目前我没有使用，可以考虑如果效果有提升的话融入进去。

# 摘要
当前的应用中期望使用推荐策略来推荐最适合的结果来改变用户行为，最大化推荐收益，如极大化点击率或者网页浏览时长等。但由于推荐系统改变了用户的行为方式，使得用户的日志行为和原本的用户自然行为有了比较大的差异。简单的在用户日志上训练和评估推荐策略，产生的结果和用户的自然行为产生比较大的误差。**为此，论文中提出一种新的领域自适应算法（domain adaptiation algorithm）[可以理解为一种新的矩阵分解策略]，从含有推荐偏差的日志数据中学习，并使用随机曝光预测推荐结果。**实验说明，得到了更好的结果。
# 介绍
近年来，在线商务已超过传统商务的增长。 因此，关于推荐系统的研究工作在相同的时间线内也在显著增长。 在线电子商务领域的主要参与者，如亚马逊，YouTube或Netflix，产品推荐功能现在是需求的关键驱动因素，在亚马逊，推荐销售约占总体销售的35％。

在推荐系统领域，出现了一类有前途的新型深度学习解决方案，并且正在展示出有希望的结果。 新方案主要有两类：学习物品向量（item embedding）和学习用户向量（user embedding），学习item embedding以优化物品相似度预测，学习user embedding以优化推荐物品预测。

这些新方法可以扩展到数百万用户和项目，并显示出优于传统方法的性能改进。 在大规模产品推荐的背景下，这些深度学习方法已成功应用于Yahoo!邮件中的广告推荐。 用于 YouTube 的视频推荐和 OpenTable 的餐馆推荐。

但这些最新的机器学习模型仍将推荐任务构建为：
- 物品与物品间/用户与物品间的距离学习问题。常使用MSE和AUC曲线来衡量结果。
- 推荐物品预测(next item prediction)问题。常使用Precision@K和标准累计收益NDCG(normalized discounted cumulative gain)来衡量。

这两类方案的模型中，都没有对**推荐的内在干预性**进行建模。在建模时，不仅应该对用户行为进行建模，而且应该考虑目前的推荐结果对其的影响。 

**该文中提出了一种标准矩阵分解方法的简单修改**。该方法利用**少量随机推荐结果样本**来创建用户和产品表示。该方法中用户和项目对之间的关联成对距离比传统方法中的 ITE(individual treatment effect) 更好。

# 相关工作
1. 倾向评分方法（IPS的相关介绍）
2. 使用领域适应和迁移学习方法的因果推断
3. 因果推荐（Causal Recommendations）

# 符号定义和数学推导
输入：一个来自用户集合 X 的用户 ui，ui∈X
推荐系统行为：针对某个用户显示某个产品的推荐策略 πx ，与每个用户有关
输出：一个来自物品集合 P 的可能推荐物品 pj，pj∈P

随机策略 πx 将用户 ui 与产品 pj 的推荐关联的概率与每个用户 ui 和产品 pj 相关联：
$$p_j ∼ π_x(·|u_i)  $$


定义 rij 为向用户 ui 推荐产品 pj 的真实奖励：
$$r_{ij}= r(·|u_i,p_j)$$

在例子中，rij是二元结果，例如点击/不点击，销售/不销售。 我们假设奖励rij根据ui和pj根据未知的条件分布r分布。

定义 yij 为记录策略 πx 下用户 ui 和 产品 pj 键对的观察奖励：
$$y_{ij}= r_{ij}π_x(p_j|u_i)$$

定义 看到用户 ui 的概率服从分布 p(X)：
$$u_i  ∼  p(\chi)$$

定义 用户ui ,物品 pj, 策略 πx 下的Reward（奖励）值：
$$R^{π_x}_{ij}=y_{ij}p(u_i)=r_{ij}π_x(p_j|u_i)p(u_i) $$

则 与策略 πx 相关联的Reward值 Rπx 等于通过使用相关的个性化产品暴露概率在所有传入用户中收集的奖励的总和：
$$R^{π_x}=\sum_{ij}R^{π_x}_{ij}=\sum_{ij}y_{ij}p(u_i) =\sum_{ij} r_{ij}π_x(p_j|u_i)p(u_i) $$

定义 用户ui ,物品 pj, 策略 πx 下的 ITE 值为其奖励与控制策略 πc 的奖励间的差异：
$$ITE^{π_x}_{ij}= R^{π_x}_{ij}− R^{π_{base}}_{ij}$$

则策略 π* 下 ，ITE总和为：
$$ITE^{π_x} =\sum_{ij}ITE^{π_x}_{ij}$$

定义在策略 π* 下 ，ITE总和最高，则 π* 为：
$$π^∗= arg \max_{π_x}\{ITE^{π_x} \}$$

其中，为找到最优策略 π* ,使用反响倾向评分法 Inverse Propensity Scoring (IPS) 来预测不可观测的 rij：

$$\hat{r}_{ij} ≈ \frac{y_{ij}}{π_x(p_j|u_i)} $$

# 损失函数
对于随机曝光的数据（treatment dataset），学习目标是：
$$ L_t=\sum_{(i,j,y_{ij}∈S_t)}l^t_{ij}=L(UΘ_t,Y_t)+Ω(Θ_t) $$
可以看出，通过用户嵌入 U 和物品嵌入 Θt 计算的内积得到用户点击概率，就是简单的FM模型。

而含有偏差的数据（control dataset），也是类似的：
$$L_c=L(UΘ_c,Y_c)+Ω(Θ_c)+Ω(Θ_t−Θ_c)$$
和之前相比只是物品嵌入Θc不同，其中两个物品嵌入之间的联系，用一个超参来控制它们的差异。

文章提出把俩个任务放在一起，得到联合损失方程（我称为物品损失函数吧）:
$$L^{prod}_{CausE}= \underbrace{L(UΘ_t,Y_t)+ Ω(Θ_t)}_{treatment\ task\ loss}+ \underbrace{L(UΘ_c,Y_c) + Ω(Θ_c)}_{control\ task\ loss}+\underbrace{Ω(Θ_t−Θ_c)}_{regularizer\ between\ tasks}$$
上标 prod 表示是在固定用户 user 下对于物品 products 的损失函数，也可以尝试固定物品对用户进行类似的损失函数实验。

最终，提出了针对不同策略和用户的多任务目标损失函数（我称为理想损失函数吧）：
$$L_{CausE}= \underbrace{L(Γ_tΘ_t,Y_t)+ Ω(Γ_t,Θ_t)}_{treatment\ task\ loss}+ \underbrace{L(Γ_cΘ_c,Y_c) + Ω(Γ_c,Θ_c)}_{control\ task\ loss}+\underbrace{Ω(Γ_t−Γ_c)+Ω(Θ_t−Θ_c)}_{regularizer\ between\ tasks}$$

但实验中，作者并未使用理想损失函数，只针对物品提出策略，所以使用的是物品损失函数进行实验：)

# 实验部分

##### 任务：评测推荐得分（Estimating Treatment Rewards ）
作者给出了评价标准。
推荐是经典的转换率预测问题，文章使用了MSE和NLL作为评价标准，并且将最后的结果描述为 **提升（lift）**：
$$lift^{metric}_x= \frac{metric_x− metric_{AvgCR}}{metric_{AvgCR}}$$
AvgCR 是平均预测性能（比如测试集上的效果），metric 是MSE或 NLL，x是使用的方法。
除了提升，还使用了AUC曲线作为一个评估标准。
##### 基线：比较的方法
1.Bayesian Personalized Ranking （BPR）
数据集就由三元组 <u,i,j> 表示，该三元组的物理含义为：相对于物品“j”，用户“u”更喜欢物品“i”。
2.Supervised-Prod2Vec （SP2V）
将yij近似为用户和产品向量内积线性变换的sigmoid
$$\hat{y_{ij}}= σ(a < p_j,u_i> +b_i+ b_j+ b) $$
3.Weighted-SupervisedP2V （WS2V）
4.BanditNet (BN)
##### 数据定义
No adaptation (no) ：只使用控制数据
Blended adaptation (blend) ：混合使用数据
Test-only adaptation (test) ：只用治疗数据（随机曝光的数据）
Product-level adaptation (prod) ：基于St样本为每个产品构建单独的向量
##### 数据集
movielens 和 netflix
使用的数据划分：
70% 训练集（60%的control data，10%的treatment data）
10% 验证集（control data）
20% 测试集（treatment data）
![实验结果](https://upload-images.jianshu.io/upload_images/6802002-78bd0383a4005b54.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
# 总结
1. 理想很美好，但对于如何获取 St （随机曝光的数据集）来进行实验，在论文和代码中都未提及。
2. 论文中通过考虑用户自然行为来提升模型效果，这个概念是在推荐领域中融入了迁移学习的思想，很有想法。但方法中的实验好像也太简单了点，没有太多的复用价值和指导意义，不适用于推广到其它复杂的模型上。


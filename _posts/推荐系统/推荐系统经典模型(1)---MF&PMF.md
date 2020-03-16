>写在前面：
>**本系列** 主要涵盖3方面内容：
> 1.Recommend System 相关的 **经典论文/方法**
> 2.Recommend System 相关的 **最新前沿论文**
> 3.这些方法基于 python + pytorch 的 **含数据的 可运行的 复现**
# 用户-物品矩阵
用户-物品交互矩阵是Recomender System最关注的数据之一，反映了用户的真实偏好。
如图的示例中表示了用户对书籍偏好。偏好的取值范围是1分到5分，5分是最高（也就是最喜欢的）。第一个用户（行1）给第一本书（列1）的评分为4分，如果某个单元格为空，代表着用户并未对这本书作出评价。
![用户-物品矩阵](https://upload-images.jianshu.io/upload_images/6802002-685a4662066beddd.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
如果用户、物品的数量较少，且交互矩阵较为密集，每个用户相当于一个长度为$N$的 one-hot 向量（$N$为物品的数量），可以直接通过用户和物品的交互情况，去为用户进行推荐。

但用户和物品的数量级较大时，往往交互矩阵极为稀疏，简单的 one-hot 向量表示已经无法很好的表达用户的喜好，我们需要使用 **LFM (Latent Factor Model) 隐因子模型**，其中隐因子中的每维单元可以理解为一个用户喜欢一本书的隐形原因。这个你平时也会有，就那种你也说不上来，不知道为啥，但就是喜欢这本书的原因。LFM的核心思路就是求出用户和物品的隐向量。

![隐因子模型](https://upload-images.jianshu.io/upload_images/6802002-b132172196de1b7a.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

# MF(Matrix factorization)
矩阵分解(matrix factorizaiton,简称 MF)最早源于 Simon Funk 在[博客](http://sifter.org/~simon/journal/20061211.html)上公布的 Funk-SVD 算法，其基本思想是从评分矩阵$R$中学习用户和物品在低维隐空间上的表示：

将用户-物品评分矩阵$R_{ n \times m}$分解成两个矩阵，$R=U^TV$，
$U=\{U_1,U_2,...,U_{n-1},U_n\} \in \mathbb{R}^{K \times n}$,每个$U_i$为$K$维，表示用户$ i $的特征向量；
$V=\{V_1,V_2,...,V_{m-1},V_m\} \in \mathbb{R}^{K \times m}$,每个$V_j$为$K$维，表示物品$ j $的特征向量；
**损失函数为**
$$Loss_{mf} =  \sum_{i=1}^N\sum_{j=1}^M{I_{ij}(R_{ij}-U_i^TV_j)^2+\alpha_r(||U||^2_F+||V||^2_F)}$$
其中，$I_{ij}\in \{0,1\}$为指示函数，表示是否有评分；$(||U||^2_F+||V||^2_F)$为正则项（具体计算为$||A||_F^2 = \sum{a_{ij}^2}$），用于控制模型的复杂度，防止过拟合；参数$\alpha_r$用于平衡两项的贡献度。
**利用凸优化方法，偏导数为0，迭代训练**
$U^{(\tau+1)}_i=U^{(\tau)}_i-\gamma (e_{ij}V_j^{(\tau)}+ \alpha_rU^{(\tau)}_i)$
$V^{(\tau+1)}_j=V^{(\tau)}_j-\gamma (e_{ij}U_i^{(\tau)}+ \alpha_rV^{(\tau)}_i)$
# PMF(Probabilistic Matrix Factorization)
Salakhutdinov 等人从 **概率** 的角度对于上述矩阵分解模型进行解释,提出了概率矩阵分解模型(probabilistic matrix factorization,简称 PMF)。
**两个假设：**
- 用户属性$U$和物品属性$V$均为高斯分布
- 观测噪声（观测评分矩阵$R$和近似评分矩阵$\hat{R}$之差）为高斯分布

即假定用户$U_i∼N(0,\sigma^2_u)$，物品$V_j∼N(0,\sigma^2_v)$，差值$R_{ij}-U^T_iV_{j}∼N(0,\sigma^2)$，即$R_{ij}∼N(U^T_iV_{j},\sigma^2)$。由于假定所有观测评分值均相互独立,可得：
$P(U|\sigma^2_u)=\prod_{i=1}^NN(0,\sigma^2_u)$
$P(V|\sigma^2_v)=\prod_{j=1}^MN(0,\sigma^2_v)$
$P(R|U,V,\sigma^2)=\prod_{i=1}^N\prod_{j=1}^M[N(u_i^Tv_j,\sigma^2)]^{I_{ij}}$
利用贝叶斯规则（最大后验概率∝先验·似然）：
$P(\theta|X)= \frac {P(X|\theta)P(\theta)}{\int P(X|\theta)P(\theta)d\theta}∝P(X|\theta)P(\theta)$
特征矩阵 U 和 V 的后验分布可通过如下方法计算:
$$P(U,V|R,,\sigma^2,\sigma^2_u,\sigma^2_v) ∝P(R|U,V,\sigma^2)P(U|\sigma^2_u)P(V|\sigma^2_v)$$
最大化特征矩阵 U 和 V 的后验概率，等价于最小化上式的负对数。
>*高斯分布公式及其对数形式：*
>$$f(x)=\frac{1}{\sqrt{2\pi}\sigma}e^{-\frac{(x-\mu)^2}{2\sigma^2}}$$
>$$lnf(x)=-ln({\sqrt{2\pi}\sigma})-\frac{(x-\mu)^2}{2\sigma^2}$$

即：
$$P(U,V)∝\frac{(R-U^TV)^2}{2\sigma^2}+\frac{||U||^2_2}{2\sigma^2_u}+\frac{||V||^2_2}{2\sigma^2_v} +C$$
![PMF](https://upload-images.jianshu.io/upload_images/6802002-dd6ecc0966901a9a.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
当超参数$(\sigma^2,\sigma^2_u,\sigma^2_v)$固定时，可以得到如下目标函数：
$$Loss_{pmf}=\frac{1}{2}\sum_{i=1}^N\sum_{j=1}^MI_{ij}(R_{ij}-U_i^TV_j)^2+\frac{\lambda_u}{2}\sum_{i=1}^N||U_i||^2_2+\frac{\lambda_v}{2}\sum_{j=1}^M||V_j||^2_2$$
为计算方便,常常假定超参数$\sigma^2_u,\sigma^2_v$相同,则上述目标函数等价于传统矩阵分解模型MF。
PMF是从概率角度很好的解释了MF。

# 代码实现
用movielens-1m数据集实现了PMF
实现见：[https://github.com/weberrr/recsys_model](https://github.com/weberrr/recsys_model)

这里主要说明一下 `pmf.py` 中的迭代更新部分：
```
                # Compute Error
                predicts = np.sum(np.multiply(
                    self.U[batch_users_id, :], self.V[batch_items_id, :]), axis=1)
                errors = predicts - train_set[shuffled_order[batch_idx], 2]
                
                # Compute Gradients
                U_grad = np.multiply(
                    errors[:, np.newaxis], self.V[batch_items_id, :])+self._lambda*self.U[batch_users_id, :]
                V_grad = np.multiply(
                    errors[:, np.newaxis], self.U[batch_users_id, :])+self._lambda*self.V[batch_items_id, :]

                # find same element to update
                U_2_update = np.zeros((num_user, self.num_feature))
                V_2_update = np.zeros((num_item, self.num_feature))
                for i in range(self.batch_size):
                    U_2_update[batch_users_id[i], :] = U_grad[i, :]
                    V_2_update[batch_items_id[i], :] = V_grad[i, :]

                # Update with epsilon
                self.U = self.U - self.epsilon * U_2_update
                self.V = self.V - self.epsilon * V_2_update
```

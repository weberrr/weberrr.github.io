# VAE(Variational Auto-Encoder)(2014)
关于VAE的介绍，建议去看苏建林的VAE讲解，写的非常好，**看完第一篇基本就可以理解VAE的原理了**。如果有时间可以阅读剩下两篇的深入分析。建议多读几遍，方便你形成一个清晰的VAE认识。
>苏剑林. (2018, Mar 18). 《变分自编码器（一）：原来是这么一回事 》[Blog post]. Retrieved from [https://www.spaces.ac.cn/archives/5253](https://www.spaces.ac.cn/archives/5253)
>苏剑林. (2018, Mar 28). 《变分自编码器（二）：从贝叶斯观点出发 》[Blog post]. Retrieved from [https://kexue.fm/archives/5343](https://kexue.fm/archives/5343)
>苏剑林. (2018, Apr 03). 《变分自编码器（三）：这样做为什么能成？》[Blog post]. Retrieved from [https://kexue.fm/archives/5383](https://kexue.fm/archives/5383)


简单来说，变分自编码器(VAE)是一种主要用于数据生成的自编码器的变体。当作为生成模型时，首先利用数据训练变分自编码器，然后只使用变分自编码器的解码部分，自动生成与训练数据类似的输出。

VAE假设后验分布$p(Z|X)～N(0,1)$。给定一个真实的样本$X_k$，假设存在**专属于$X_k$的分布**$p(Z|X_k)$，我们用两个神经网络$u_k=f_1(X_k)$，$log \sigma^2 _k=f_1(X_k)$去寻找专属于$X_k$的后验分布$p(Z|X_k)$的均值和方差。（选择拟合$log \sigma^2 _k$是因为$ \sigma^2 _k$总是非负的，用网络拟合一个非负值需要加激活函数，而$log \sigma^2 _k$可正可负。）

因为$p(Z)=p(Z|X)p(X)$，从这个专属的分布中采样一个$Z_k$出来，经过生成器得到$\hat{X_k}=g(Z_k)$，最小化$D(\hat{X_k},X_k)$即为网络的目标。

但最小化$D(\hat{X_k},X_k)$的拟合过程会使得方差尽可能接近0，使得随机性下降，退化为普通的AutoEncoder，这样VAE特有的生成模型就不存在了。因此，VAE让$p(Z|X)$尽可能去看齐$N(0,1)$，防止了噪声（方差）为0，保证了模型的生成能力。根据$$p(Z)=∑_Xp(Z|X)p(X)=∑_XN(0,I)p(X)=N(0,I)∑_Xp(X)=N(0,I)$$
得到了$p(Z)～N(0,I)$，即可以放心的从N(0,I)中采样去生成图像了。只使用变分自编码器的解码部分，自动生成与训练数据类似的输出。

![VAE](https://upload-images.jianshu.io/upload_images/6802002-a365c964dbd910f7.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

如何让所有的$p(Z|X)$都向$N(0,I)$看齐呢？使用KL散度：
$$KL(N(u,\sigma^2)||N(0,I))=L_{u,\sigma^2}=\frac{1}{2}\sum_{i=1}^d(μ^2_{(i)}+σ^2_{(i)}−logσ^2_{(i)}−1)$$
其中，d 为隐向量的维度。
>**相对熵又称KL散度**，如果我们对于同一个随机变量 x 有两个单独的概率分布 P(x) 和 Q(x)，我们可以使用 KL 散度（Kullback-Leibler (KL) divergence）来衡量这两个分布的差异。
>$$KL(p(x)||q(x))=\int p(x)ln \frac{p(x)}{q(x)}dx$$

VAE的优点总结：
- 在传统自编码器的隐层表达上增加一个对隐变量的约束,提出了一种将概率模型和神经网络结构结合的方法
- 使编码器产生的隐层表达满足正态分布,能够更好的生成和输入近似的数据
---

# CVAE(Collaborative Variational AutoEncoder)(2017)
论文：[Collaborative Variational Autoencoder for Recommender Systems
](http://xueshu.baidu.com/usercenter/paper/show?paperid=56fe358614c8cef3fdf9b746c09e0eb2&site=xueshu_se&hitarticle=1)，KDD，2017

CVAE的核心思想是：CVAE = PMF + VAE（与CTR/CDL类似的耦合结构）
如果仔细阅读过《[推荐系统实践(3)---CDL]()》中介绍的CDL的paper，你会发现CVAE的通篇描述以及文章结构甚至都和CDL很像，Loss构成相近，参数的迭代更新方式也是一样的。CVAE结构如图所示。
![CVAE](https://upload-images.jianshu.io/upload_images/6802002-496c9a29f6b31ac9.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

### 原理介绍
与PMF相同，作者定义向量分布如下：
**User 隐向量：** $u_i～N(0,\lambda^{-1}_uI_k)$
**Item 协同信息隐向量：**$v_j^*～N(0,\lambda^{-1}_vI_k)$
**VAE部分：**VAE网络用于从 item 的内容信息中提取隐向量。这里需要重点说明一下，作者在文章中将**item 内容信息隐向量**定义为$z_j$，有$z_j～N(mu_j,\sigma^2_j)$，其中$mu_j,log\sigma^2_j$是由 VAE的 encoder 网络生成。实际上，$z_j = mu_j + z_{std}*\sigma_j$，如下图所示，中间态的$z_j$是由一个标准正态分布重参数获得的，因而$z_j$是服从正态分布的重采样，用于docoder中生成数据，不能表示 item 的内容信息隐向量。
![重参数](https://upload-images.jianshu.io/upload_images/6802002-611d03a86647da4d.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/640)
看了作者的代码实现，在实验中，作者的 encoder 网络生成的**item 内容信息隐向量**应该是$mu_j$，这验证了我的想法。
**item 隐向量：** 作者表示为$v_j=v_j^*+z_j$，但我觉得表示为$v_j=v_j^*+mu_j$更为合适，下文沿用这种写法。
**Prediction：**$R_{ij}^*=u_i^Tv_j=u_i^T(v_j^*+mu_j)$

### 损失函数

$Loss_{cvae}=Loss_{pmf}+loss_{vae}$
$Loss_{pmf}=\frac{1}{2}\sum_{i=1}^N\sum_{j=1}^MC_{ij}(R_{ij}-u_i^Tv_j)^2+\frac{\lambda_u}{2}\sum_{i=1}^N||u_i||^2_2+\frac{\lambda_v}{2}\sum_{j=1}^M||v_j-mu_j||^2_2$
$Loss_{vae}=\sum_{j=1}^M KL(N(mu_j,\sigma^2_j)||N(0,1))+CEloss+ \frac{\lambda_w}{2}\sum_{l}(||W_l||^2_F+||b_l||^2_2)$ 
其中 CEloss 是item的内容信息 encoder前 和 docoder后 的差值，类似于交叉熵的计算形式。

### 迭代形式
作者与CTR/CDL相同，采用 **EM算法**。
**E**xpectation：固定$U,V$，利用VAE网络的bp更新$V_{theta}$(即$mu$)
**M**aximization：固定$V_{theta}$，利用块坐标下降法更新$U,V$
值得一提的是，在实现时，作者并没有使用$v_j=v_j^*+mu_j$的形式，而是使用$v_j$，并希望$v_j$去逼近其内容向量$mu_j$。
$u_i ←(VC_iV^T+\lambda_uI_k)^{-1}VC_iR_i$
$v_j←(UC_jU^T+\lambda_vI_k)^{-1}(UC_jR_j+\lambda_v\theta_j)$

### 改进之处
1.CVAE直接对标的模型就是CDL，作者这样描述CDL的问题：
>*denoising autoencoders(DAEs) have in fact no Bayesian nature and the denoising scheme of the DAEs is in fact not from a probabilistic perspective but rether frequentist perspective.*

即CDL的降噪方案不是来自概率分布而是频率分布，因此很难进行贝叶斯推理。并且噪声添加方式需要考虑内容信息的形式，例如文本内容的加噪方案可能不是图像内容的良好加噪方案。
2.作者使用 **逐层预训练** 的VAE模型来有效表示内容信息的隐向量，这一点是非常值得学习的，好的预训练，会让实验结果差非常多。

# 代码实现
用pytorch实现了CVAE，并用citeulike-a数据集做了实验。
[https://github.com/weberrr/recsys_model](https://github.com/weberrr/recsys_model)

主要说一下实现中的model.fit()函数：
```
    def fit(self, train_users, test_users, train_items, item_side_info):
        side_input = self.initialize(train_users, train_items, item_side_info)
        for epoch in range(self.max_epoch):
            loss, side_latent = self.e_step(side_input)
            self.V[:] = side_latent.clone().detach().numpy()
            recall = self.m_step(train_users, train_items, test_users)
            print("Epoch:{}, Loss:{}, Recall:{}".format(epoch, loss, recall))
```
e_step中固定U,V，更新V_theta
m_step中固定V_theta，更新U,V
![实验结果截图](https://upload-images.jianshu.io/upload_images/6802002-cdb731cb91ac2c09.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
我的VAE逐层预训练做的比较潦草，效果比作者的低约0.02左右

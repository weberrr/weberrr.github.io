本文叙述结构为：
- AE(AutoEncoder)
- DAE(Denosing AutoEncoder)
- SDAE(Stack Denosing AutoEncoder)
- CDL(Collaborate Deep Learning)
- 代码实现
---
# AE(AutoEncoder) 
自动编码器是一种数据压缩算法，是利用反向传播算法使得输出值等于输入值的神经网络。其算法包括编码阶段（encoder）和解码（decoder）阶段。
**编码器：**这部分能将输入压缩成潜在空间表征，可以用编码函数h=f(x)表示。
**解码器：**这部分能重构来自潜在空间表征的输入，可以用解码函数r=g(h)表示。
从自编码器获得有用特征的一种方法是，限制h的维度使其小于输入x，这种情况下称作有损自编码器。通过训练有损表征，使得自编码器能学习到数据中最重要的特征。
目前，自编码器的应用主要有两个方面，第一是**数据去噪**，第二是**为进行可视化而降维**。设置合适的维度和稀疏约束，自编码器可以学习到比PCA等技术更有意思的数据投影。
![AE](https://upload-images.jianshu.io/upload_images/6802002-faf84d324b52821a.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
# DAE(Denoising AutoEncoder)(2008)
自编码器真正关心的是**隐藏层的特征表达**，一个好的表达能够捕获输入信号的稳定结构，以该目的为出发出现了降噪自动编码器。
DAE的直观解释：有点类似人体的感官系统，比如人眼看物体时，如果物体某一小部分被遮住了，人依然能够将其识别出来。
DAE首先对干净的输入信号随机加入噪声产生一个受损的信号。然后将受损信号送入传统的自动编码器中，使其重建回原来的无损信号。
![DAE](https://upload-images.jianshu.io/upload_images/6802002-8ecd0eb9aa5c36b8.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
降噪自编码器与传统的自编码器的主要区别在于:
- 降噪自编码器通过人为的增加噪声使模型获得鲁棒性的特征表达
- 避免使隐层单元学习一个传统自编码器中没有意义的恒等函数
- 缺陷在于每次进行网络训练之前，都需要对干净输入信号添加噪声，以获得它的损坏信号，就增加了模型的处理时间。
# SDAE(Stacked Denoising Auto-Encoders)(2008)
参考深度置信网络的方法，将降噪自编码器进行堆叠可以构造成堆叠降噪自编码器。
**SDAE逐层贪婪训练：**每层自编码层都单独进行非监督训练，以最小化输入（输入为前一层的隐层输出）与重构结果之间的误差为训练目标。前K层训练好了，就可以训练K+1层，因为已经前向传播求出K层的输出，再用K层的输出当作K+1的输入训练K+1层。
![SDAE](https://upload-images.jianshu.io/upload_images/6802002-cc160fc9fb9a71a2.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

一旦SDAE训练完成, 其高层的特征就可以用做传统的监督算法的输入。SDAE并不能进行模式识别，因为它只是一个特征提取器，并不具有分类功能。当然，也可以在最顶层添加一层logistic regression layer（softmax层），然后使用带label的数据来进一步对网络进行微调（fine-tuning），即用样本进行有监督训练，便得到具有分类功能的SDAE。

![SDAE](https://upload-images.jianshu.io/upload_images/6802002-905f5442fdd138f2.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

堆叠降噪自编码器与降噪自编码器的区别在于:
- 堆叠降噪自编码器采用了降噪编码器的编码器作为基础单元,并且使用其训练方法进行预训练
- 降噪自动编码器是**无监督学习(自监督)**的一种方法,而降噪自编码器是一种有监督方法.

# CDL(Collaborative Deep Learning)(2015)
[推荐系统实践(2)---CTR]() 中介绍了CTR（Collaborative Topic Regression）模型。CTR模型将 Item的内容信息（side information）很好融合到了 vj 中，缓解矩阵稀疏带来的冷启动问题；同时，其融合内容信息的思想和耦合式模型结构，给后续研究者提供了思路，在此基础上人们开展了各种拓展和应用。CDL是其中的优秀代表之一。

论文：[Collaborative Deep Learning for Recommender Systems](http://xueshu.baidu.com/usercenter/paper/show?paperid=ca73b49b0c772b70c2be12eb1de04779&site=xueshu_se&hitarticle=1)，SIGKDD，2015

>CDL论文原话： 
>*CTR is an appealing method in that it produces promising and interpretable results.Nevertheless, **the latent representation learned is often not effective enough especially when the auxiliary information is very sparse**. It is this representation learning problem that we will focus on in this paper.*
>CDL主要就是解决CTR模型中辅助信息稀疏时的有效表示问题。

![CDL](https://upload-images.jianshu.io/upload_images/6802002-dd7bdad16a2e3d23.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

有了CTR的介绍基础，我们可以很容易看懂CDL的模型结构。类似于CTR，CDL = PMF + SDAE。利用SDAE去有效学习 vj 的词向量。
$$v_j =\epsilon_j + X^T_{L/2,j∗}$$
其中，$X^T_{L/2,j∗}$为SDAE部分的中间层的隐向量表示。
**因为下篇要介绍的CVAE与本篇的模型结构、思想类似，所以这篇的论文细节和原理介绍相对简略，看懂下篇基本就可以看懂CDL的结构。**

### 损失函数
$$Loss_{cdl}=∑_i∑_j\frac{I_{ij}}2(R_{ij}−u^T_iv_j)^2+\frac{λ_u}2∑_i||u_i||^2_2+\frac{λ_v}2∑_j||v_j−f_e(X_{0,j*,}W^+)^T||^2_2+\frac{λ_n}2∑_j||f_r(X_{0,j*,}W^+)-X_{c,j*}||^2_2+\frac{λ_w}2∑_l(||W_l||^2_2+||b_l||^2_2)$$
可以看出，损失函数分几个部分：
-  $∑_i∑_j\frac{I_{ij}}2(R_{ij}−u^T_iv_j)^2$ 部分，表示预测误差
- $\frac{λ_u}2∑_i||u_i||^2_2$部分，表示用户向量 ui 的正则化项
- $\frac{λ_v}2∑_j||v_j−f_e(X_{0,j*,}W^+)^T||^2$部分，表示物品向量 vj 和 物品 j 的内容向量尽可能拟合
- $\frac{λ_n}2∑_j||f_r(X_{0,j*,}W^+)-X_{c,j*}||^2_2$部分，表示SDAE的网络损失
- $\frac{λ_w}2∑_l(||W_l||^2_2+||b_l||^2_2)$部分，表示SDAE网络的参数的正则化项

### 迭代方式
与CTR相同，采用 **EM算法**。
**E**xpectation：固定$U,V$，利用SDAE网络的bp更新$\theta$
**M**aximization：固定$\theta$，利用块坐标下降法更新$U,V$

# 代码实现
因CDL和CVAE类似，在下一篇中给出了CVAE的模型复现。
用citeulike-a数据集实现了CVAE。
[https://github.com/weberrr/recsys_model](https://github.com/weberrr/recsys_model)

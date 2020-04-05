# 1.Batch Normalization 详解
链接：https://zhuanlan.zhihu.com/p/34879333

笔记：
### BN是什么？
深层的DNN容易发生internal covariate shift问题（上层网络需要不停调整来适应输入的变化，学习率低，同时训练过程容易陷入梯度饱和区），batch normailzation通过将网络层的输入进行normalization，将输入的均值和方差固定在一定范围内，减少了ICS。
### BN表达式
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
### BN的优点
1. 保证每层输入稳定，加速模型收敛
2. 参数不那么敏感，使网络更加稳定
3. BN有一定的正则化效果：每个batch的均值方差不同，随机增加噪音

# 2. 生成模型和判别模型

生成式模型举例：

利用生成模型是根据山羊的特征首先学习出一个山羊的模型，然后根据绵羊的特征学习出一个绵羊的模型，然后从这只羊中提取特征，放到山羊模型中看概率是多少，在放到绵羊模型中看概率是多少，哪个大就是哪个。   

常见的生成模型：隐马尔科夫模型、朴素贝叶斯模型、高斯混合模型、 LDA、 Restricted Boltzmann Machine 等。

判别式模型举例：

要确定一个羊是山羊还是绵羊，用判别模型的方法是从历史数据中学习到模型，然后通过提取这只羊的特征来预测出这只羊是山羊的概率，是绵羊的概率。

常见的判别模型有线性回归、对数回归、线性判别分析、支持向量机、 boosting

# 2. 召回策略演化

https://zhuanlan.zhihu.com/p/97821040

# 3. 排序策略演化

https://zhuanlan.zhihu.com/p/100019681

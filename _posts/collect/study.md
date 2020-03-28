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
论文：[Deep Neural Networks for YouTube (2016)](https://cseweb.ucsd.edu/classes/fa17/cse291-b/reading/p191-covington.pdf)

# System Review
Youtube作为全球最大的UGC的视频网站，需要在百万量级的视频规模下进行个性化推荐。由于候选视频集合过大，考虑online系统延迟问题，不宜用复杂网络直接进行推荐，所以Youtube采取了两层深度网络完成整个推荐过程：
![Youtube推荐系统](https://upload-images.jianshu.io/upload_images/6802002-64a017e4de0b2e7c.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
之所以把推荐系统划分成Generation和Ranking两个阶段，主要是从**性能方面**考虑的。Generation阶段面临的是百万级视频，单个视频的性能开销必须很小；而Ranking阶段的算法则非常消耗资源，不可能对所有视频都算一遍。

第一层是 **Candidate Generation Model**。完成候选视频的快速筛选，这一步候选视频集合由百万降低到了百的量级。
第二层是 **Ranking Model**。完成几百个候选视频的精排。
图中的 **other candidate sources** 指的是2010年Youtube发在Recsys上的之前的推荐算法下的候选视频。

# Candidate Generation Model
模型结构如图所示
![candidate generation model](https://upload-images.jianshu.io/upload_images/6802002-b2c8e408ff7ccae6.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/640)
#### 模型输入
**embedded videos&search tokens**
我们自底而上看这个网络，最底层的输入是用户观看过的video的embedding向量，以及搜索词的embedding向量。至于这个embedding向量是怎么生成的，作者的原话是这样的
>*Inspired by continuous bag of words language models, we learn high dimensional embeddings for each video in a fixed vocabulary and feed these embeddings into a feedforward neural network.*

所以，**作者是先用word2vec方法对video和search token做了embedding之后再作为输入的**。对于这些历史观看视频序列，可根据重要性和时间进行加权平均，来得到**固定长度的watch vector**。
*这里我觉得还有一种可以尝试的思路是加一个embedding层跟上面的DNN一起训练，但两种方法孰优孰劣需要根据结果来吹。*

**其他特征：**
- **人口统计学信息**：性别（binary）、年龄（continuous）、地域等
- **其他上下文信息**：设备、登录状态等
` 这些连续或离散特征都被归一化为[0,1]， 和watch vector以及search vector做拼接（concat） `
- **视频上传时间信息（example age）**：该特征表示视频被上传之后的时间。

#### 模型输出
- 该用户 i 的向量 user_vector ui
- 百万视频中每个视频的向量 video_vector vj (即百万视频softmax过程的权值)

>问：sofamax多分类问题中，Youtube的candidate video有百万之巨，意味着有几百万个分类，这必然会影响训练效果和速度，如何改进？

答：
We rely on a technique to sample negative classes from the background distribution ("candidate sampling") and then correct for this sampling via importance weighting.
简单说就是进行了负采样（negative sampling）并用importance weighting的方法对采样进行calibration。

>问：在candidate generation model的serving过程中，YouTube为什么不直接采用训练时的model进行预测，而是采用了一种最近邻搜索的方法？

答：
这个问题的答案是一个经典的工程和学术做trade-off的结果，在model serving过程中对几百万个候选集逐一跑一遍模型的时间开销显然太大了，因此在通过candidate generation model得到user 和 video的embedding之后，通过最近邻搜索的方法的效率高很多。我们甚至不用把任何model inference的过程搬上服务器，只需要把user embedding和video embedding存到redis或者内存中就好了。

# Ranking Model
既然得到了几百个候选集合，下一步就是利用ranking模型进行精排序。ranking模型结构如图所示：
![ranking model](https://upload-images.jianshu.io/upload_images/6802002-d64dcb413aa71d5e.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
#### 模型输入
- **impression video ID embedding**：当前要计算的video的embedding
- **watched video IDs average embedding**：用户观看过的最后N个视频embedding的average pooling
`video embedding（video vector）由candidate generation model传入`
- **language embedding**：用户语言的embedding和当前视频语言的embedding
- **time since last watch**：自上次观看同channel视频的时间
- **previous impressions**：该视频已经被曝光给该用户的次数。一定程度上引入了exploration的思想，避免同一个视频持续对同一用户进行无效曝光。尽量增加用户没看过的新视频的曝光可能性。
>针对某些特征，比如previous impressions，为什么要进行开方和平方处理后，当作三个特征输入模型？

答：**这是很简单有效的工程经验，引入了特征的非线性。**就好像Inception中使用1x1,3x3,5x5三个卷积一样，让模型自己去学习哪个更好。从YouTube这篇文章的效果反馈来看，提升了其模型的离线准确度。
#### 模型输出
我们的目标是**预测期望观看时长**。通过时长来对视频排序。

**训练输出**
有点击的为正样本，有PV无点击的为负样本，正样本需要根据观看时长进行加权。训练阶段网络最后一层用的是 weighted logistic regression。
其中，权重$w_i = \begin{cases} T_i ,&\text{positive sample}\\ 1 ,&\text{negative sample} \end{cases}$，Ti为样本 i 的观看时长
引入变量Odds（机会比），其定义为：$Odds=\frac{p}{1-p}$
令$ln(Odds)=\theta^Tx$，求解p，得：
$$p=\frac{1}{1+e^{-\theta^Tx}}=sigmod(\theta^Tx)$$
加权的LR即为对上式中的 $p$ 加权。
Weighted LR的特点是，正样本权重w的加入会让正样本发生的几率变成原来的w倍，也就是说样本 i 的Odds变成了下面的式子：
$$Odds(i)=\frac{w_ip}{1-w_ip}$$
由于在视频推荐场景中，用户打开一个视频的概率 $p$ 往往是一个很小的值，因此上式可以继续简化：
$$Odds(i)=\frac{w_ip}{1-w_ip}≈w_ip=T_ip=E(T_i)$$
Weighted LR使用用户观看时长 Ti 作为权重，使得对应的Odds(i)表示的就是用户观看时长的期望E(Ti)。

**线上输出**
我们看图可知线上输出为：$e^{Wx+b}$。下面推导它。
如果对Odds取自然对数，再让ln(Odds)等于一个线性回归函数，那么就得到了下面的等式。
$$ln(Odds)=ln(\frac{p}{1-p})=\theta_0+\theta_1x_1+\theta_2x_2+...+\theta_nx_n=\theta^Tx$$

Odds即可解为：
$$Odds = e^{\theta^Tx}=YouTubeServingFunction$$
说明线上观察的值，即为Odds(i)，等于E(Ti)---期望观看时长。
**总结**
- 线上的$e^{Wx+b}$这一指数形式计算的是Weighted LR的Odds；
- Weighted LR使用用户观看时长作为权重，使得对应的Odds表示的就是用户观看时长的期望；
- 因此，Model Serving过程中$e^{Wx+b}$计算的正是观看时长的期望。

# 其他工程细节
>在进行video embedding的时候，为什么要直接把大量长尾的video直接用0向量代替？

答：这又是一次工程和算法的trade-off，把大量长尾的video截断掉，主要还是为了节省online serving中宝贵的内存资源。当然从模型角度讲，低频video的embedding的准确性不佳是另一个“截断掉也不那么可惜”的理由。

>在对训练集的预处理过程中，YouTube没有采用原始的用户日志，而是对每个用户提取等数量的训练样本，这是为什么？

答：为了减少高度活跃用户对于loss的过度影响。
>YouTube为什么不采取类似RNN的Sequence model，而是完全摒弃了用户观看历史的时序特征，把用户最近的浏览历史等同看待，这不会损失有效信息吗？

答：这个原因应该是YouTube工程师的“经验之谈”，如果过多考虑时序的影响，用户的推荐结果将过多受最近观看或搜索的一个视频的影响。YouTube给出一个例子，如果用户刚搜索过“tayer swift”，你就把用户主页的推荐结果大部分变成tayer swift有关的视频，这其实是非常差的体验。为了综合考虑之前多次搜索和观看的信息，YouTube丢掉了时序信息，讲用户近期的历史纪录等同看待。

>在处理测试集的时候，YouTube为什么不采用经典的随机留一法（random holdout），而是一定要把用户最近的一次观看行为作为测试集？

答：这个问题比较好回答，只留最后一次观看行为做测试集主要是为了避免引入future information，产生与事实不符的数据穿越。

>在确定优化目标的时候，YouTube为什么不采用经典的CTR，或者播放率（Play Rate），而是采用了每次曝光预期播放时间（expected watch time per impression）作为优化目标？

答：这个问题从模型角度出发，是因为 watch time更能反应用户的真实兴趣，从商业模型角度出发，因为watch time越长，YouTube获得的广告收益越多。而且增加用户的watch time也更符合一个视频网站的长期利益和用户粘性。



# 参考链接
[Deep Neural Network for YouTube Recommendation论文精读](https://zhuanlan.zhihu.com/p/25343518)
[YouTube深度学习推荐系统的十大工程问题](https://zhuanlan.zhihu.com/p/52504407)
[重读Youtube深度学习推荐系统论文，字字珠玑，惊为神文](https://zhuanlan.zhihu.com/p/52169807)
[论文笔记：Deep neural networks for YouTube recommendations](https://blog.csdn.net/xiongjiezk/article/details/73445835)

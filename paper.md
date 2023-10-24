# 方法介绍

在这一部分 我们将详细介绍两部分内容：情绪抽取和时序建模。

![图片1.png](图片1.png)
    Figure1. 模型全体框架。从左到右分别是从每日的新闻标题中提取情绪价值、两种情绪信息特征EWSA和WASA以及捕捉长期情绪价值的股价预测模型Romember。

## 问题定义
在股票预测这个问题中，我们方法的输入分别有L天股市结束时刻股价和情绪信息，输出是未来T天的闭市股价数值。对于输入部分的股价时间序列 可以表示为$ X = (x_1, x_2, ..., x_L), X \in R^{L}$, 其中每一项$x_i$代表序列中第$i$天的股价。输入部分的情绪信息序列可以表示为$ E = (e_1, e_2, ..., e_L), E \in R^{L}$，$e_i$表示序列中第$i$天的情绪特征。输出可以表示$Y = (y_{L+1}, y_{L+2}, ..., y_{L+T} ), Y \in R^{T}$, 其中$y_i$表示预测第$i$天的股价数值。鉴于情感对于股价的影响的即时性，我们的一般考虑$T = 1$的情况。

## 情感抽取
这一部分我们将讨论如何处理大量的新闻文本信息和如何组织这些新闻中的情绪信息，对应图1中的Sentiment Extraction部分，也是近一步解释模型输入中的$e_i$是如何获得的。在第$i$天，我们可以在新闻媒体上获得大量的新闻文本$News_i = \{News_i^{1},News_i^{2}, ..., News_i^{M_i}\}$,$M_i$是第$i$日获取到的新闻文本数量。对于每一篇新闻文本$News_i^{m}$，我们使用提前finetune好的情感分析文本模型 RoBERTa，得到当前新闻的情感倾向特征$Sent_i^{m}$。$Sent_i^{m}$取值可以是-1、0、1，分别对应情绪的积极、中性和消极。每篇新闻的情绪特征具体可以表示为如下：
$$
    Sent_i^{m} = RoBERTa(News_i^{m})\\ 
    Sent_i^{m} \in \{ -1, 0, 1\}
$$
我们还需要把当天的情绪值做一个聚合，所以当天的情绪特征表示为：
$$
    Sent_i = \frac{1}{M}\sum_{m = 1}^{M} Sent_i^{m}
$$
因为股市存在停市的特殊情况，这种情况下市场情绪依旧是每日产生的，并且会影响到开市当天的股价走向。同时市场情绪确实也会在一个时间范围内对市场价格产生影响，所以我们需要统计当天前一个窗口期内的股市情绪。假设窗口大小为W，是一个可以设置的超参数，那么我们可以通过取平均的方式获取到窗口期内的市场情绪，称为WASA(Weighted avg sentiment analysis), 即：
$$
   WASA_i = \frac{1}{W} \sum_{w = 1}^{W} Sent_{i - w}
$$
考虑到时间间隔越久情绪价值越小，本作还提出指数加权的情绪特征，具体是:
$$
    EWSA_i = (1 - \alpha) * EWSA_{i-1} +  \alpha * Sent_i \\
    0 \lt \alpha \leq 1 \\
    EWSA_0 = 0
$$
WASA_i和EWSA_i都可以用问题定义中的$e_i$表示。至此 我们得到了情绪特征时序，与股价时序结合之后，我们可以得到下一阶段的输入$F$：
$$
    F = concate([X,E]) \\
    F \in R^{L×2}
$$
其中任意一项$f_i = concate([x_i, e_i])$。实验中，我们通过去掉$e_i$,或者切换$e_i$的类型来分析情绪特征的作用。

## Romember：情绪长期建模
首先，我们选择Transformer架构为基础做股价与情绪的长期建模，这一点是因为理论上Transformer比LSTM，CNN等神经网络有更长的依赖关联能力。但是Transformer架构中的MultiHeadSelfAttention存在随着长度变长$O(L^2)$复杂度的问题。所以我们首先借鉴了Informer中的ProbAttention将复杂度降低到$O(Lln(L))$的程度。其次我们还注意到ProbAttention中对距离信息的缺失，于是又使用另外一种旋转位置编码来弥补这一不足。这一部分对应图1中的右边Romember部分。
### ProbAttention
在Transformer架构中，我们一般把输入转变为 query, key, value 三个部分，然后通过以下的方式，算出query对key的关联程度以及最后用加权和的方式组合起来关联的部分。具体公式为：
$$
    A(Q, K, V) = Softmax(\frac{QK^{T}}{\sqrt{d}})V
$$
其中$ Q \in R^{L_Q×d}, K \in R^{L_K×d}, V \in R^{L_V ×d}$。为了深入讨论，我们令$q_i, k_i, v_i$ 代表 $Q,K,V$中的一项。 对于Q每一项$q_i$都会计算其与所有$k_i$的内积值$a_{ij} = q_ik_j^{T}$，这也是复杂度是$O(L^2)$的原因。
ProbAttention是在Informer这一篇时序预测的工作中提出的，主要是在attention增加了两步计算。1）随机选出$ln(L_k)$个$k_j$再与所有$q_i$求内积。2） 根据上一步结果选出$ln(L_q)$个$q_i$，并且与所有$k_j$求内积得到attenion分数矩阵。上面两步的算法复杂度总的来说就是$O(Llin(L))$。
### Rotational Position Embedding

我们知道Transformer和Informer的位置信息也很有存在的必要，他们能够表明每一个数据点所在的位置。但是他们在attenion计算中，却无法反应两个位置的相对信息，主要是因为绝对位置编码经过多层线性变换之后的点积已经不再等于两个绝对位置$m与n$之差的函数了。在ProbAttention中，经过筛选后的$q_i$和$k_j$因为不一定相邻，那么相对位置信息就会显得更加重要。因为$k_j$分别和$q_i$和$q_{i+1}$求内积时，$q_i$与$q_{i+1}$只是位置相邻，但实际上中间可能被筛掉了未知长度的数据，但是在$k_j$看来 他们有可能是紧邻的，也有可能是逆序的。无疑相对位置信息的缺失，在时序任务中一定会造成精度的下降。所以我们必须为每个$q_i、k_i$附上足以表明相对位置的位置信息。RoPE就是一种能够反映相对位置的编码方式，目前已经在自然语言处理领域中的生成任务中应用广泛(可以参考Llama、ChatGLM)。具体就是对于任意的$q_i$，$k_j$，都乘以一个旋转位置编码$R_{\theta,pos}^{d}$。
$$
    rq_i = R_{\Theta,i}^{d}q_i \\
    rk_j = R_{\Theta,i}^{d}k_j
$$
增加了旋转位置编码之后的内积可以表示为:
$$
\begin{aligned}
    rq_irk_j^{T} &= (R_{\Theta,i}^{d}q_i)(R_{\Theta,i}^{d}k_j)^{T} \\
    &= q_iR_{\Theta, j-i}^{d} k_j^{T}
\end{aligned}
$$
这上式中$R_{\theta, j-i}^{d}$正好能够表示位置$i$与$j$的相对距离。它这么神奇的原因是其内部组成是由正余弦的旋转矩阵组成的。可以表示为:
$$
R_{\Theta, i}^d=\left(\begin{array}{ccccccc}
\cos i \theta_1 & -\sin i \theta_1 & 0 & 0 & \cdots & 0 & 0 \\
\sin i \theta_1 & \cos i \theta_1 & 0 & 0 & \cdots & 0 & 0 \\
0 & 0 & \cos i \theta_2 & -\sin i \theta_2 & \cdots & 0 & 0 \\
0 & 0 & \sin i \theta_2 & \cos i \theta_2 & \cdots & 0 & 0 \\
\vdots & \vdots & \vdots & \vdots & \ddots & \vdots & \vdots \\
0 & 0 & 0 & 0 & \cdots & \cos i \theta_{d / 2} & -\sin i \theta_{d / 2} \\
0 & 0 & 0 & 0 & \cdots & \sin i \theta_{d / 2} & \cos i \theta_{d / 2}
\end{array}\right)
$$
其中$\Theta$是超参数，可以通过以下方式提前计算好:
$$
\Theta = \{\theta_i = 10000^{-2(i - 1)/d}, i \in [1,2, ..., d/2]\}
$$
我们利用旋转位置编码对于ProbAttention中的位置缺失进行改进，能更好的见面长期时间序列，也更有利于我们引入更长期的市场情绪变化规律。
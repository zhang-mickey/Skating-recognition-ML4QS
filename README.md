# fall detection

Detecting falls accurately is vital for preventing injuries and providing timely assistance. Falls and non-falls can be characterized by changes in motion, specifically in terms of range, angle, and speed. A typical fall involves a sudden movement towards the ground, culminating in a vertical shock.

One of the primary challenges we face is collecting and capturing fall data accurately.

To capture these  patterns, we use three devices. One phone is placed on the central part of the waist, aligning with the body’s center of gravity, while the other is attached to the dominant foot to record all data associated with foot movements. This combination allows us to comprehensively capture the dynamics of a fall.

We leverages five types of sensors to capture the essence of these motions accurately. Additionally, we incorporate physiological data, specifically heart rate, to gain better insight into the nature of falling behavior, although it may introduce  missing values.



Since I am a rookie, due to the lack of experience in this sports, I fall consistently,  so that we were able to collect the data   in a situation close
to  real-life fall scenarios (when falls appear unintentionally)

Falls can occur in various directions, but they often exhibit similar patterns. To streamline our data collection and analysis process, we have chosen to focus on falls in one direction only. 

How  we define the window size? Our sample rate is 70HZ, Based on relevant research and experimental validation, we choose the sliding window 50

## Multiclass
Target: predict the activity

**power strokes**

**breaking**

**turns**

**jumps**

**curves**

**Heart rate**
捕捉 重心变化 
**accelerometer**

**gyroscope**

**megnetometer**

**Location GPS**

## Sensors 数据收集流程
Obtainment of a sufficient quantity of long-term, reliable and portable recordings
![image](https://github.com/zhang-mickey/Skating-recognition-ML4QS/assets/145342600/4315d2f6-0a97-44c8-8aa2-d4f2a228b4bc)
### Placement of sensor
![sensor_schematic](https://github.com/zhang-mickey/Skating-recognition-ML4QS/assets/145342600/48eb9da2-876f-4f89-8c7a-5cd7ba1acd21)


# Data preprocessing
## 时序数据
### 时间戳的转换
### 数据重采样（修改时间频率）
取平均值
### 异常值处理、数据平滑处理

### Chauvenets Criterion
assume the data to follow the normal distribution
#### Mixture model

Assuming we have K distributions to describe our data  

#### Kalman filter
状态估计算法
#### lowpass filter


### topic  model
无监督

 topic modeling generally generates fewer features compared to a bag of words approach with 1-grams, as it condenses the information into a smaller number of high-level topics rather than individual words.****
#### LDA 隐含狄利克雷分布
它将每个文档视为多个主题的混合，而每个主题又是多个词的混合
#### Fourier transformation

Tries to decompose a temporal sequence of measurements with some form of periodicity into a set of sinusoid functions of different frequencies.  


The data obtained
by the inertial sensor are divided into smaller data segments of
a predetermined size, namely, the data window. Obviously, it is
realistic to use sliding windows with a specific overlap rate to
avoid data sample loss.

Therefore, we resort to annotating in terms of event-specific
time intervals to suit the training process. Although there is
nonuniformity in the data, we balance the collected data generated from each source, avoiding inaccurate behavior for identifying falls. 、

## EDA
### 数据分布、数据相关性

### 特征提取/相关性分析

#### Pearson coefficient 
simple but could ignore more complex dependencies,e.g. multiple features,that only together exhibit some predicitve power 
#### PCA
![image](https://github.com/zhang-mickey/Skating-recognition-ML4QS/assets/145342600/a7c5d54b-8ddc-430a-aac5-b2ab833cf7ae)

### Frequency Domain

#### Fourier Transformations

#### amplitude
take the **frequency with highest amplitude**, this gives us an indication of the most important frequency in the windows under consideration

Frequency weighted signal average 

power spectral entropy: determines whether there are one or a few discrete frequency standing out of all others





### forward selection 

## Person Level Distance Metrics
![image](https://github.com/zhang-mickey/Skating-recognition-ML4QS/assets/145342600/bd7f40db-bde9-431d-8466-8093af039ac8)

### Temporal distancce metrics
#### cross-correlation coefficient
allows for time series that are shifted

#### DTW
take into account that there is a difference in speed between different time series

### k-medoids
Instead of assigning cluster centers that are averages over the points belonging to the cluster, the k-medoids algorithm selects points from the dataset as cluster centers. 
This solves our previous problem we have identified with the person level data and also works well with dynamic time warping. 

### subspace clustering 
we split the range of each variable up into ε distinct intervals  

we define a cluster as a maximal set of connected dense units.

common face (i.e. when they share a border of a range on one dimension and have the same ranges on the others)

### silhouette score

a:the average distance of a point to the other points in its cluster

b:the average distance to the points in the cluster closest 
## PAC 概率近似正确
Basically we call a hypothesis se PAC learnable, if given enough training exapmles we can approximate the out- of -sample error arbitrarily well by the in-sample error

## VC dimension
For a perceptron, which creates a hyperplane using a linear combination of feature values, the VC dimension is directly related to the number of features, d, used to describe the input space. Specifically, the VC dimension of a perceptron is d + 1.
In d-dimensional space, a perceptron can shatter d + 1 points. This is because a set of d + 1 points can be positioned in such a way that every possible binary labeling of these points can be separated by some hyperplane.


A hypothesis set with an infinite VC dimension:

we cannot provide guarantees on the different between te in sample and out of sample error according to the learning theory
## SVM
find a hyperplane that maximizes the distance between two classes 
## Naive Bayes
under the assumption that the attributes are conditionally independent.  
## decision tree
Regularization Hyperparameters

为避免过拟合，需要在训练时限制决策树的自由度，这被称作正则化。正则化超参数跟算法有关，但一般情况下至少可以限制决策树的最大深度

### L1 norm

### L2 norm

## bagging(reduce variance)

## boosting(reduce bias)


## 多目标优化 Pareto Front
核心在于如何平衡各个目标之间的关系。一种常见的方法是使用帕累托支配关系（Pareto Dominance）来比较解的优劣。如果一个解在所有目标上都不差于另一个解，且至少在一个目标上严格优于另一个解，则称该解支配另一个解
# Time series
## 自回归模型 Autoregressive Model
不依赖于别的解释变量，只依赖于自己过去的历史值，故称为自回归；如果依赖过去最近的p个历史值，称阶数为p，记为AR(p)模型

![image](https://github.com/zhang-mickey/Skating-recognition-ML4QS/assets/145342600/cda69320-af79-4c2c-9f0d-7ee1c037b7f7)

线性回归与自回归的比较

两种回归方法都假设过去的变量与未来的值呈线性关系。线性回归根据同一时间段内的多个自变量预测结果。同时，自回归仅使用一种变量类型，但将其扩展到几个点以预测未来的结果。例如，您可以使用线性回归根据天气、交通量和步行速度来预测通勤时间。或者，自回归模型使用您过去的通勤时间来估计今天的到达时间。
## 移动平均模型
当前时间点的值等于过去若干个时间点的预测误差的回归；预测误差=模型预测值-真实值；如果序列依赖过去最近的q个历史预测误差值，称阶数为q，记为MA(q)模型

MA(1)相当于一个无穷阶的AR模型。同样的，对于任意的q，MA(q)均可以找到一个AR模型与之对应。因此，我们可以得到，时间序列数据归根到底，是可以用统一用AR模型来表示的

那么，为什么还要MA模型呢？如果只有AR模型，那么一些时间序列必然会需要很高的阶数p来刻画。阶数p，就是待估参数的个数。待估参数越多，需要付出的参数估计代价就越大，所以我们当然希望参数个数越少越好。因此我们自然希望能够用低阶的MA模型来替换高阶的AR模型；
## ARMA模型 AutoRegressive Moving Average Model
由自回归部分和移动平均部分组成，用于捕捉时间序列中的自相关性和季节性

AR(p)可以看成ARMA(p, 0)

任何一个时间序列都可以用纯粹的AR模型来刻画。但是偏自相关系数无穷阶后都不收敛于0，说明只能用一个高阶的AR来解释。但这样的话，阶数太高，待估参数太多，我们就不开心了。所以我们对这个高阶AR模型做分解，分解出一个低阶的AR模型和另一个特殊的高阶AR模型，其中分解出来的高阶AR模型恰好等价于一个低阶的MA模型。于是我们就可以用低阶的AR模型和低阶的MA模型来描述这个时间序列了，这就是ARMA模型
## ARIMA
因为时间序列分析要求平稳性，不平稳的序列需要通过一定手段转化为平稳序列，一般采用的手段是差分；d表示差分的阶数，t时刻的值减去t-1时刻的值，得到新的时间序列称为1阶差分序列；1阶差分序列的1阶差分序列称为2阶差分序列，以此类推


## RNN
RNN 在内部设计上存在一个严重的问题：由于网络一次只能处理一个时间步长，后一步必须等前一步处理完才能进行运算
### Backpropagation
如果在输出层得不到期望的输出值，则取输出与期望的误差的平方和作为目标函数，转入反向传播，逐层求出目标函数对各神经元权值的偏导数，构成目标函数对权值向量的梯量，作为修改权值的依据.

用链式法则对每层迭代计算梯度
high computational overhead, and the probability of getting stuck in a local minimum.

反向传播算法缺乏仿生学方面的理论依据，显然生物神经网络中并不存在反向传播这样的数学算法
### echo state network
![image](https://github.com/zhang-mickey/Skating-recognition-ML4QS/assets/145342600/2c3fad59-5949-4b2b-9454-74b8f383a410)

关键部分是随机生成的稀疏储备池（Reservoir Computing）。

ESN的训练, 只需要利用线性回归方法训练输出权值, 输入权值和储备池权值根据特定的要求随机生成

ESN的这种训练方式能够保证权值的全局最优, 克服了基于梯度的递归神经网络计算效率低、训练方法复杂以及容易陷入局部最优等问题
## LSTM
![image](https://github.com/zhang-mickey/Skating-recognition-ML4QS/assets/145342600/bd8bdcf2-39b1-45fa-ad5e-8b56f3cc9e3d)
### Forget Gate

## TCN
Embedding的主要目的是将时序数据映射到一个稠密的连续向量空间中，使得相似的语义信息在该向量空间中也能够彼此接近


## 感受野
![image](https://github.com/zhang-mickey/Skating-recognition-ML4QS/assets/145342600/6db030e7-6357-4ac0-81d5-43f49586bcc4)

### causal convolution

### dilated convolution
![image](https://github.com/zhang-mickey/Skating-recognition-ML4QS/assets/145342600/538dc45a-f540-47c7-b9f6-fcd721e0468d)

### 空洞卷积

## Reinforcement Learning

### MDP
### discount factor
0: only care about the immediate reward

1:future rewards are equally important as the current reward 

### SARSA on policy
using the same policy 

The estimates of the value of an action or state are updated by considering the same action selection mechanism in the next state
### Q-learning off policy
the action with the highest Q-value is always selected in the next state 

![image](https://github.com/zhang-mickey/Skating-recognition-ML4QS/assets/145342600/7f129f68-0411-43f0-89ba-d69ef3297c09)

#### eligibility traces资格迹 


### U-tree to handle a continuous state space
dynamically discretizes the state space by means of a state tree

We build the tree by starting with a single node (i.e. a leaf): all continuous states are mapped to one discrete state. 

For each leaf, we investigate whether a split could be beneficial. Hereto, we iterate over all attributes Xi and sort the data according to the values of that attribute. We consider a split at each point in the ordered list of values and test for a significant difference between the Q-values in the two resulting sets using a Kolmogorov Smirnov test. We store both the p-value and the split criterion of the split with the lowest p-value



#### apple watch导出数据
Apple Health 导出为 CSV 格式数据

Apple Health XML to CSV Converter

https://www.ericwolter.com/projects/apple-health-export/

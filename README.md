# fall detection
Falls and nonfalls can be regarded as changes in motion in terms of range, angle and speed.
## Kalman filter
状态估计算法
# 
Apple Health 导出为 CSV 格式数据

Apple Health XML to CSV Converter

https://www.ericwolter.com/projects/apple-health-export/


# Data preprocessing
The data obtained
by the inertial sensor are divided into smaller data segments of
a predetermined size, namely, the data window. Obviously, it is
realistic to use sliding windows with a specific overlap rate to
avoid data sample loss.

Therefore, we resort to annotating in terms of event-specific
time intervals to suit the training process. Although there is
nonuniformity in the data, we balance the collected data generated from each source, avoiding inaccurate behavior for identifying falls. 
## Sensors
Obtainment of a sufficient quantity of long-term, reliable and portable recordings
![image](https://github.com/zhang-mickey/Skating-recognition-ML4QS/assets/145342600/4315d2f6-0a97-44c8-8aa2-d4f2a228b4bc)
### Placement of sensor
![sensor_schematic](https://github.com/zhang-mickey/Skating-recognition-ML4QS/assets/145342600/48eb9da2-876f-4f89-8c7a-5cd7ba1acd21)


**Heart rate**

**accelerometer**

![image](https://github.com/zhang-mickey/ML4QS/assets/145342600/8537a095-fa96-46c6-ba79-a7b9eefd7afa)

**gyroscope**

**megnetometer**

**Location GPS**

## Frequency & time period  

## Multiclass
Target: predict the activity

**power strokes**

**breaking**

**turns**

**jumps**

**curves**
# Time series
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

## TCN
Embedding的主要目的是将时序数据映射到一个稠密的连续向量空间中，使得相似的语义信息在该向量空间中也能够彼此接近


## 感受野
![image](https://github.com/zhang-mickey/Skating-recognition-ML4QS/assets/145342600/6db030e7-6357-4ac0-81d5-43f49586bcc4)

### causal convolution

### dilated convolution
![image](https://github.com/zhang-mickey/Skating-recognition-ML4QS/assets/145342600/538dc45a-f540-47c7-b9f6-fcd721e0468d)

### 空洞卷积

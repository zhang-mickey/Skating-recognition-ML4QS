# fall detection

## Kalman filter
状态估计算法
# 
Apple Health 导出为 CSV 格式数据

Apple Health XML to CSV Converter

https://www.ericwolter.com/projects/apple-health-export/


# 分帧
假设每帧只有一个活动，为每个帧打标签  
## Sensors
![image](https://github.com/zhang-mickey/ML4QS/assets/145342600/495df10a-2cac-45f4-ae9a-e60077f35431)

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
### Backpropagation
如果在输出层得不到期望的输出值，则取输出与期望的误差的平方和作为目标函数，转入反向传播，逐层求出目标函数对各神经元权值的偏导数，构成目标函数对权值向量的梯量，作为修改权值的依据.

用链式法则对每层迭代计算梯度

反向传播算法缺乏仿生学方面的理论依据，显然生物神经网络中并不存在反向传播这样的数学算法
## LSTM

## TCN
Embedding的主要目的是将时序数据映射到一个稠密的连续向量空间中，使得相似的语义信息在该向量空间中也能够彼此接近
## 感受域

### causal convolution

### dilated convolution
### 空洞卷积

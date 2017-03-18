
# coding: utf-8

# In[1]:

import numpy as np
#本项目实现简单神经网络模型

#非线性函数
#前向传播：直接返回sigmoid激活函数
#反向传播：对sigmoid函数求倒数,即x*(1-x)
def nonlin(x,deriv=False):
    if (deriv == True):
        return x*(1-x)
    return 1/(1+np.exp(-x))

#输入样本
x = np.array([[0,0,1],
             [0,1,1],
             [1,0,1],
             [1,1,1]])

#0代表一类，1代表一类
y = np.array([[0],
             [1],
             [1],
             [0]])


np.random.seed(1)

#初始化权重参数,构造一个3行4列的权重参数矩阵
w0 = 2*np.random.random((3,4))-1
w1 = 2*np.random.random((4,1))-1

for j in xrange(60000):
    #输入层
    l0 = x 
    #中间层
    l1 = nonlin(np.dot(l0,w0))
    #输出层
    l2 = nonlin(np.dot(l1,w1))
    
    #计算输出与我们预定的结果的差距,如果l2_error很大，说明误差很大，
    #需要对W进行较大的调整
    l2_error = y-l2
    
    #每10000次打印一下
    if (j % 10000) == 0:
        print "Error:" + str(np.mean(np.abs(l2_error)))

    #求l2的梯度
    l2_delta = l2_error * nonlin(l2,deriv=True)
    
    #算出l1对追踪结果误差的影响,w1.T是w1的梯度
    l1_error = l2_delta.dot(w1.T)

    #求l1的梯度
    l1_delta = l1_error * nonlin(l1,deriv=True)
    
    #对W0，W1进行修正
    w1 += l1.T.dot(l2_delta)
    w0 += l0.T.dot(l1_delta)


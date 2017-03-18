
# coding: utf-8

# In[1]:


'''
Created on 2017年3月18日
#本项利用RNN递归神经网络实现二进制加法
@author: qiujiahao
'''
import numpy as np
import copy
#激活函数
def sigmoid(x):
    output = 1/(1+np.exp(-x))
    return output
#sigmoid函数的倒数值
def sigmoid_output_to_derivative(output):
    return output*(1-output)

int2binary = {}
binary_dim = 8

largest_number = pow(2, binary_dim)

#建立一个hash表把0-255每个数的二进制码建立映射
binary = np.unpackbits(
    np.array([range(largest_number)],dtype=np.uint8).T,axis=1)

for i in range(largest_number):
    #将对应的10进制数转化为二进制
    int2binary[i]=binary[i]
    
#输入参数
#学习速率
alpha = 0.1
#输入的维度:2个2进制的数
input_dim = 2
#中间层
hidden_dim = 16
output_dim = 1

#初始化权重，h代表中间层的循环结构
synapse_0 = 2*np.random.random((input_dim,hidden_dim))  -1
synapse_1 = 2*np.random.random((hidden_dim,output_dim)) -1
synapse_h = 2*np.random.random((hidden_dim,hidden_dim)) -1

#反向传播完了，要去更新参数，这里定义一个和权重一样维度的矩阵
#用来存储更新参数
synapse_0_update = np.zeros_like(synapse_0)
synapse_1_update = np.zeros_like(synapse_1)
synapse_h_update = np.zeros_like(synapse_h)

#开始进行前向传播
for j in range(10000):
    
    #随机的找一个整数值，除以2是方便c不要超过largest_number
    a_int = np.random.randint(largest_number/2)
    a = int2binary[a_int]
    
    b_int = np.random.randint(largest_number/2)
    b = int2binary[b_int]  
    
    #真实的值
    c_int = a_int + b_int 
    c = int2binary[c_int]
    
    #d用来装预测值
    d = np.zeros_like(c)
    
    overallError = 0
    
    layer_2_deltas = list()
    layer_1_values = list()
    #为了第一次迭代的时候能取到前一次的结果,此处加一个全0的值
    layer_1_values.append(np.zeros(hidden_dim))

    #本二进制数一共有8位，依次计算
    for position in range(binary_dim):
        x = np.array([[a[binary_dim-position-1],b[binary_dim-position-1]]])
        y = np.array([[c[binary_dim-position-1]]]).T
        
        #当前阶段的值乘以权重synapse_0，并加上上一阶段的值乘以synapse_h
        layer_1 = sigmoid(np.dot(x,synapse_0)+np.dot(layer_1_values[-1],synapse_h))
        
        #输出层并没有循环
        layer_2 = sigmoid(np.dot(layer_1,synapse_1))
        
        #误差
        layer_2_error = y - layer_2
        #计算L2层的权重对输出的影响,sigmoid_output_to_derivative是计算sigmod函数的倒数值
        layer_2_deltas.append((layer_2_error)*sigmoid_output_to_derivative(layer_2))
        overallError +=np.abs(layer_2_error[0])
        
        #实际输出
        d[binary_dim-position-1] = np.round(layer_2[0][0])
        #复制保存l1层的输出
        layer_1_values.append(copy.deepcopy(layer_1))
    
    #用于将来保存循环权重的影响
    future_layer_1_delta=np.zeros(hidden_dim) 
    
    #进行反向传播计算
    for position in range(binary_dim):
        x = np.array([[a[position],b[position]]])
        #取出前向传播时保存的值
        layer_1 = layer_1_values[-position-1]
        #取出前一阶段的值
        prev_layer_1 = layer_1_values[-position-2]
        #l2层的误差
        layer_2_delta = layer_2_deltas[-position-1]
        
        #l1层的误差，注意future_layer_1_delta.dot(synapse_h.T)计算的是循环部分的影响
        layer_1_delta = (future_layer_1_delta.dot(synapse_h.T) + layer_2_delta.dot(synapse_1.T)) * sigmoid_output_to_derivative(layer_1) 
        
        #w1是前一层传下来的delta值乘以当前层的输出值得到w1所需要更新的大小
        synapse_1_update += np.atleast_2d(layer_1).T.dot(layer_2_delta)
        #prev_layer_1是隐层计算出来的值
        synapse_h_update += np.atleast_2d(prev_layer_1).T.dot(layer_1_delta)

        synapse_0_update += x.T.dot(layer_1_delta)
        
        future_layer_1_delta = layer_1_delta
    
    # 我们已经完成了所有的反向传播，可以更新几个转换矩阵了。并把更新矩阵变量清零
    #alpha代表更新的力度
    synapse_0 += synapse_0_update * alpha
    synapse_1 += synapse_1_update * alpha
    synapse_h += synapse_h_update * alpha

    synapse_0_update *= 0
    synapse_1_update *= 0
    synapse_h_update *= 0
    
    # print out progress
    if(j % 1000 == 0):
        print("Error:" + str(overallError))
        print("Pred:" + str(d))
        print("True:" + str(c))
        out = 0
        for index,x in enumerate(reversed(d)):
            out += x*pow(2,index)
        print(str(a_int) + " + " + str(b_int) + " = " + str(out))
        print("------------")  


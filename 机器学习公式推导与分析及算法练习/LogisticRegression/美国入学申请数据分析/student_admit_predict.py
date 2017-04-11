#coding:utf-8
'''
Created on Apr 9, 2017

@author: ubuntu

'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
admissions = pd.read_csv("admissions.csv")

#先使用线性回归分析
from sklearn.linear_model import LinearRegression
mode1 = LinearRegression()
mode1.fit(admissions[['gre','gpa']],admissions['admit'])
admit_prediction = mode1.predict(admissions[['gre','gpa']]) 
plt.xlabel("gpa")
plt.ylabel("admit_prediction")
plt.scatter(admissions["gpa"],admit_prediction)
plt.show()
'''
在上图中可见，有些预测结果小于0，而这明显是不对的，
因为预测结果应该只能为0或者1，我们现在需要获取一个介于0和1之间的概率
'''
#使用逻辑回归
def logit(x):
    return np.exp(x)/(1+np.exp(x))
#在-6到6之间等差产生50个数
t = np.linspace(-6,6,50,dtype=float)
y_logist = logit(t)
plt.plot(t,y_logist,label="logistic")
plt.ylabel("Probability")
plt.xlabel("t")
plt.title("Logistic Function")
plt.show()

'''
假设t为线性回归，那么logit(t)就是逻辑回归，本次我们使用sklearn中的库函数
'''
from sklearn.linear_model import LogisticRegression
#对数据进行重新排序
admissions = admissions.loc[np.random.permutation(admissions.index)]
#前700条数据作为训练，其余用来测试
num_train = 500
data_train = admissions[:num_train]
data_test  = admissions[num_train:]

logis_model = LogisticRegression()
logis_model.fit(data_train[["gpa","gre"]], data_train["admit"])

filt_test = logis_model.predict_proba(data_test[["gpa","gre"]])[:,1]
plt.scatter(data_test['gre'],filt_test)
plt.xlabel('gre')
plt.ylabel('probability ')
plt.show()

#评估模型
#现在假设只要录取概率大于0.5的就能录取，计算一下这个模型的准确性
#predict()函数会自动把阀值设置为0.5

#准确率
predicted =  logis_model.predict(data_test[["gpa","gre"]])
accuracy=(predicted == data_test["admit"]).mean()
print "accuracy:%f"%accuracy

#ROC曲线
from sklearn.metrics import roc_auc_score,roc_curve

test_predict=logis_model.predict_proba(data_test[["gpa","gre"]])[:,1]
train_predict=logis_model.predict_proba(data_train[["gpa","gre"]])[:,1]
#AUC(Area under Curve)：Roc曲线下的面积，介于0.1和1之间。Auc作为数值可以直观的评价分类器的好坏，值越大越好
auc_test=roc_auc_score(data_test["admit"],test_predict)
auc_train=roc_auc_score(data_train["admit"],train_predict)
print "test auc score:%f"%auc_test
print "train auc score:%f"%auc_train

roc_train=roc_curve(data_test["admit"],test_predict)
roc_test=roc_curve(data_train["admit"],train_predict)
plt.plot(roc_train[0],roc_train[1])
plt.plot(roc_test[0],roc_test[1])
plt.xlabel("TPR")
plt.ylabel('FPR')
plt.show()

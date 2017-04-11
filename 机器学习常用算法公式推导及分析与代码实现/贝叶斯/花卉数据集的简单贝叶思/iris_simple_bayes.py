#coding:utf-8
'''
Created on Apr 10, 2017

@author: ubuntu
'''
# 我们直接取iris数据集.其实就是根据花的各种数据特征，判定是什么花
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
iris=datasets.load_iris()

#假定这些特征是符合高斯分布的
gnb=GaussianNB()
predict=gnb.fit(iris.data, iris.target).predict(iris.data)
accuracy = (predict==iris.target).mean()
print accuracy
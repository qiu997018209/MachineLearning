
# coding: utf-8

# In[1]:


'''
Created on 2017年3月19日

@author: qiujiahao
'''


import numpy as np
#贝叶斯实质上也是一个二分类

def loadDataSet():#数据格式
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]#1 侮辱性文字 ， 0 代表正常言论
    return postingList,classVec

def createVocabList(dataset):#创建词汇表
    #set基本功能包括关系测试和消除重复元素
    vocabset = set([])
    
    for document in dataset:
        #取出dataset里的每一行，进行去重和合并操作，最后得到一个一维的数据
        vocabset = vocabset | set(document) #创建并集,实现去重的功能
    #将set数据转化为列表    
    return list(vocabset)

def setOfWord2VecMN(vocabList,inputSet):
    #根据词汇表，将句子转化为向量
    #classVec中0的类别一条向量，类别为1的一条向量
    #生成一个全0的，和vocabList一样维度的列表，相同的index对应着对应的单词，
    #不同的值代表这个单词出现的次数
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec
#训练:trainMatrix是转换好的向量,trainCategory是对应的类别
def trainNB0(trainMatrix,trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory)/float(numTrainDocs)
    p0Num = np.ones(numWords);p1Num = np.ones(numWords)#计算频数初始化为1
    p0Denom = 2.0;p1Denom = 2.0                  #即拉普拉斯平滑
    for i in range(numTrainDocs):
        if trainCategory[i]==1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    #单词出现在第0类的概率和第1类的概率
    p1Vect = np.log(p1Num/p1Denom)#注意
    p0Vect = np.log(p0Num/p0Denom)#注意
    return p0Vect,p1Vect,pAbusive

def classifyNB(vec2Classify,p0Vec,p1Vec,pClass1):
    p1 = sum(vec2Classify * p1Vec) + np.log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + np.log(1-pClass1)
    #在向量0和向量1中的概率高，那么就属于哪一类
    if p1 > p0:
        return 1
    else:
        return 0

def testingNB():#流程展示
    listOPosts,listClasses = loadDataSet()#加载数据
    myVocabList = createVocabList(listOPosts)#建立词汇表
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWord2VecMN(myVocabList,postinDoc))
    p0V,p1V,pAb = trainNB0(trainMat,listClasses)#训练
    #测试，指定一条新输入的语句
    testEntry = ['love','my','dalmation']
    thisDoc = setOfWord2VecMN(myVocabList,testEntry)
    print testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb)
    
    testEntry = ['stupid','dog']
    thisDoc = setOfWord2VecMN(myVocabList,testEntry)
    print testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb)
    
if __name__ == "__main__":
    testingNB()


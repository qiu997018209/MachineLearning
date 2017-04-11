#coding:utf-8
'''
Created on Apr 10, 2017

@author: ubuntu
'''
import re #正则表达式
from bs4 import BeautifulSoup  #html标签处理
import pandas as pd

def review_to_wordlist(review):
    '''
    把IMDB的评论转成词序列
    '''
    #去掉HTML标签，拿到内容
    review_text = BeautifulSoup(review).getText()
    #用正则表达式获取符合规范的部分：^ 符号表示必须从文本开始处匹配
    review_text = re.sub("[^a-zA-z]"," ",review_text)
    #小写化所有的词，并转成词list
    words = review_text.lower().split()
    return words
#“header= 0”表示该文件的第一行包含列名称,“delimiter=\t”表示字段由\t分割, quoting=3告诉Python忽略双引号
train=pd.read_csv("labeledTrainData.tsv",header=0,delimiter="\t",quoting=3)
test=pd.read_csv("testData.tsv",header=0,delimiter="\t",quoting=3)
# 取出情感标签，positive/褒 或者 negative/贬
y_train = train['sentiment']
# 将训练和测试数据都转成词list
train_data = []
for i in xrange(0,len(train["review"])):
        #列表中的每一个元素用空格连接起来
        train_data.append(" ".join(review_to_wordlist(train["review"][i])))
test_data = []
for i in xrange(0,len(test["review"])):
        train_data.append(" ".join(review_to_wordlist(test["review"][i])))    
'''    
特征处理
TF-IDF简介:TF-IDF倾向于过滤掉常见的词语，保留重要的词语,字词的重要性随着它在文件中出现的次数成正比增加，但同时会随着它在语料库中出现的频率成反比下降
TF:term frequency词频
IDF:inverse document frequency逆向文件频率
词频 (TF) 是一词语出现的次数除以该文件的总词语数。
假如一篇文件的总词语数是100个，而词语“母牛”出现了3次，那么“母牛”一词在该文件中的词频就是3/100=0.03。
一个计算文件频率 (DF) 的方法是测定有多少份文件出现过“母牛”一词，然后除以文件集里包含的文件总数。
所以，如果“母牛”一词在1,000份文件出现过，而文件总数是10,000,000份的话，其逆向文件频率就是 log(10,000,000 / 1,000)=4。
最后的TF-IDF的分数为0.03 * 4=0.12
'''
from sklearn.feature_extraction.text import TfidfTransformer as TFIV
# 初始化TFIV对象，去停用词，加2元语言模型
#让一个词语的概率依赖于它前面一个词语。我们将这种模型称作bigram（2-gram，二元语言模型
#各参数介绍http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
tfv = TFIV(min_df=3,  max_features=None, strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}', ngram_range=(1, 2), use_idf=1,smooth_idf=1,sublinear_tf=1, stop_words = 'english')
# 合并训练和测试集以便进行TFIDF向量化操作
X_all = train_data + test_data
len_train = len(train_data)

# 这一步有点慢，去喝杯茶刷会儿微博知乎歇会儿...
tfv.fit(X_all)
#
X_all = tfv.transform(X_all)
# 恢复成训练集和测试集部分
X = X_all[:len_train] 
X_test = X_all[len_train:]


# 多项式朴素贝叶斯
from sklearn.naive_bayes import MultinomialNB as MNB

model_NB = MNB()
model_NB.fit(X, y_train) #特征数据直接灌进来
MNB(alpha=1.0, class_prior=None, fit_prior=True)

from sklearn.cross_validation import cross_val_score
import numpy as np

print "多项式贝叶斯分类器20折交叉验证得分: ", np.mean(cross_val_score(model_NB, X, y_train, cv=20, scoring='roc_auc'))
# 多项式贝叶斯分类器20折交叉验证得分: 0.950837239
'''
# 折腾一下逻辑回归，恩
from sklearn.linear_model import LogisticRegression as LR
from sklearn.grid_search import GridSearchCV
# 设定grid search的参数
grid_values = {'C':[30]}  
# 设定打分为roc_auc
model_LR = GridSearchCV(LR(penalty = 'L2', dual = True, random_state = 0), grid_values, scoring = 'roc_auc', cv = 20) 
# 数据灌进来
model_LR.fit(X,y_train)
# 20折交叉验证，开始漫长的等待...
GridSearchCV(cv=20, estimator=LR(C=1.0, class_weight=None, dual=True, 
             fit_intercept=True, intercept_scaling=1, penalty='L2', random_state=0, tol=0.0001),
        fit_params={}, iid=True, loss_func=None, n_jobs=1,
        param_grid={'C': [30]}, pre_dispatch='2*n_jobs', refit=True,
        score_func=None, scoring='roc_auc', verbose=0)
#输出结果
print model_LR.grid_scores_
'''

'''
在这些问题中，朴素贝叶斯能取得和逻辑回归相近的成绩，但是训练速度远快于逻辑回归，真正的直接和高效。
'''















    
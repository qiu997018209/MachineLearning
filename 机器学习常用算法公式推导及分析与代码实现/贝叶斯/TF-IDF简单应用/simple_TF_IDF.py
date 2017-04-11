#coding:utf-8
'''
Created on Apr 11, 2017

@author: ubuntu
'''
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

if __name__ == "__main__":
    corpus=["我 来到 北京 清华大学",#第一类文本切词后的结果，词之间以空格隔开
        "他 来到 了 网易 杭研 大厦",#第二类文本的切词结果
        "小明 硕士 毕业 与 中国 科学院",#第三类文本的切词结果
        "我 爱 北京 天安门"]#第四类文本的切词结果
    vectorizer=CountVectorizer()#该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频
    transformer=TfidfTransformer()#该类会统计每个词语的tf-idf权值
    vec = vectorizer.fit_transform(corpus)#fit_transform是将文本转为词频矩阵
    print vec
    tfidf=transformer.fit_transform(vec)#fit_transform是计算tf-idf
    print tfidf
    word=vectorizer.get_feature_names()#获取词袋模型中的所有词语
    weight=tfidf.toarray()#将tf-idf矩阵抽取出来，元素a[i][j]表示j词在i类文本中的tf-idf权重
    print weight
    for i in range(len(weight)):#打印每类文本的tf-idf词语权重，第一个for遍历所有文本，第二个for便利某一类文本下的词语权重
        print u"-------这里输出第",i,u"类文本的词语tf-idf权重------"
        for j in range(len(word)):
            print word[j],weight[i][j]
'''
-------这里输出第 0 类文本的词语tf-idf权重------
中国 0.0
北京 0.52640543361
大厦 0.0
天安门 0.0
小明 0.0
来到 0.52640543361
杭研 0.0
毕业 0.0
清华大学 0.66767854461
硕士 0.0
科学院 0.0
网易 0.0
-------这里输出第 1 类文本的词语tf-idf权重------
中国 0.0
北京 0.0
大厦 0.525472749264
天安门 0.0
小明 0.0
来到 0.414288751166
杭研 0.525472749264
毕业 0.0
清华大学 0.0
硕士 0.0
科学院 0.0
网易 0.525472749264
-------这里输出第 2 类文本的词语tf-idf权重------
中国 0.4472135955
北京 0.0
大厦 0.0
天安门 0.0
小明 0.4472135955
来到 0.0
杭研 0.0
毕业 0.4472135955
清华大学 0.0
硕士 0.4472135955
科学院 0.4472135955
网易 0.0
-------这里输出第 3 类文本的词语tf-idf权重------
中国 0.0
北京 0.61913029649
大厦 0.0
天安门 0.78528827571
小明 0.0
来到 0.0
杭研 0.0
毕业 0.0
清华大学 0.0
硕士 0.0
科学院 0.0
网易 0.0
'''





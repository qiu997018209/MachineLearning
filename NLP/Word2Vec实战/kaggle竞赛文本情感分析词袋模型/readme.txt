kaggle竞赛:https://www.kaggle.com/c/word2vec-nlp-tutorial
简介:根据用户的评论进行文本情感分析

处理流程:
1.去除html标签
2.用正则表达式去除标点
3.变成小写的分词
4.去除停用词
5.用CountVectorizer将词谱TOP5000的提取处理，转化为向量，作为特征
6.用随机森林进行模型训练,使用训练集进行预测，使用混淆矩阵进行评估效果
7.将预测结果保存
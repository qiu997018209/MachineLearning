kaggle竞赛:https://www.kaggle.com/c/word2vec-nlp-tutorial
简介:根据用户的评论进行文本情感分析

1.使用上一集目录下的“训练word2vec模型”代码训练好的word2vec模型

2.使用比赛的数据:labeledTrainData.tsv和testData.tsv

3.先导入上一步训练的模型

4.将数据进行同样的清洗

5.每一个word都可以在model里获取到一个300维的向量，将这个review里每个词的向量做一个平均，也就是说一个review得到一个300维的向量

6.使用随机森林构建模型

7.将结果保存维kaggle需要的格式
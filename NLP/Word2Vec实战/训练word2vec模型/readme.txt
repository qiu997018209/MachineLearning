kaggle竞赛:https://www.kaggle.com/c/word2vec-nlp-tutorial
简介:根据用户的评论进行文本情感分析

数据集为以上比赛的：unlabeledTrainData.tsv，共50000条评论

思路:
	1.用BeautifulSoup获取文本
	2.将整段文字分成句子
	3.对每一句话进行，利用正则表达式去掉标点，切分乘一个个小写单词，去停用词，这样就得到了一句话的词向量
	4.用gensim进行训练
	5.测试效果
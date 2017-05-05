项目简介:使用隐马可夫模型进行词性标注

1.预处理词库:使用nltk里自带的Brown词库，并在开始和结尾加上 (START, START) (END, END)

2.词统计: P(wi | ti) = count(wi, ti) / count(ti),即count(ti)代表出现t的词性的次数
  count(wi, ti)代表本处出现单词w并且词性是t的次数，P(wi | ti)代表已知词性是t单词是w的概率

3.隐层的马科夫链：P(ti | t{i-1}) = count(t{i-1}, ti) / count(t{i-1})，P(ti | t{i-1})  代表当前词性是t{i-1}，下一个词的词性是ti的概率

4.viterbi维特必算法的实现：
  对于句子里的每一个word:
	对于所有的词性:
		curr_best = max(前一个词性t{i-1}出现的概率×P(ti | t{i-1})× P(wi | ti)的概率)
		记录这个tag的前一个的最佳tag为curr_best
		记录当前viterbi的tag所对应的概率

  

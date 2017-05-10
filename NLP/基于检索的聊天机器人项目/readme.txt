1.数据:Ubuntu对话数据集，来自Ubuntu的IRC网络上的对话日志
https://drive.google.com/open?id=0B_bZck-ksdkpVEtVc1R6Y01HMWM

2.匹配:Q1 A1作为输入，匹配为1，不匹配为0

3.输入:分词之后，以词嵌入作为输入

4.损失函数:交叉熵

5.LSTM网络:Net.PNG,其中隐向量C为256*1的向量，代表问题，隐向量R代表256*1的答案向量，M代表256*256的参数矩阵，需要去学习

	   最后得到一共1*1的值，通过sigmoid转化为概率

	   注意点:图片中LSTM有2组，实际只需要一组即可，因为都是文本信息的抽取，此处2组是为了方便理解

 6.环境:Tensorflow
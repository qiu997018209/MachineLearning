#coding:utf-8
'''
Created on Apr 5, 2017

@author: ubuntu
'''
import input_data
import tensorflow as tf

mnist = input_data.read_data_sets("data", one_hot=True)

x = tf.placeholder("float",[None,784])

w = tf.Variable(tf.zeros([784,10]))

b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, w)+b)

y_ = tf.placeholder("float",[None,10])

#交叉熵
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
#因为softmax是凸函数，所以可以用此梯度下降的公式
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

init=tf.initialize_all_variables()

with tf.Session() as sess:
    
    sess.run(init)
    for i in range(1000):
        x_batch,y_batch=mnist.train.next_batch(100)
        print x_batch
        print "........"
        print y_batch
        sess.run(train_step,feed_dict={x:x_batch,y_:y_batch})
        
    correct_predict = tf.equal(tf.arg_max(y_,1), tf.arg_max(y,1))
    
    accuracy=tf.reduce_mean(tf.cast(correct_predict,'float'))
    print sess.run(accuracy,feed_dict={x:mnist.test.images,y_:mnist.test.labels})    

'''
Extracting data/train-images-idx3-ubyte.gz
Extracting data/train-labels-idx1-ubyte.gz
Extracting data/t10k-images-idx3-ubyte.gz
Extracting data/t10k-labels-idx1-ubyte.gz
0.9163
'''
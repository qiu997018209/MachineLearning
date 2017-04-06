'''
Created on Apr 4, 2017

@author: ubuntu
'''
import tensorflow as tf

sess=tf.InteractiveSession()

x = tf.Variable([1.0,2.0])
a = tf.constant([3.0,3.0])

x.initializer.run()


sub = tf.sub(x,a)
print sub.eval()

#[-2. -1.]
'''
Created on Apr 4, 2017

@author: ubuntu
'''
import tensorflow as tf

input1 = tf.constant(4.0)
input2 = tf.constant(2.0)
input3 = tf.constant(3.0)

input_add = tf.add(input2,input3)
input_mul = tf.mul(input1, input_add)

with tf.Session() as sess:
    print sess.run([input_add,input_mul])
    
#[5.0, 20.0]

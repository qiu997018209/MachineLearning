'''
Created on Apr 4, 2017

@author: ubuntu
'''
import tensorflow as tf
import numpy as np

matrix1 = tf.constant([[3.,3.]])


matrix2 = tf.constant([[2.],[2.]])

product = tf.matmul(matrix1,matrix2)

with tf.Session() as sess:
    result = sess.run([product])
    print result

#[array([[ 12.]], dtype=float32)]
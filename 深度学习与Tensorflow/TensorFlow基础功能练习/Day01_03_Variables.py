'''
Created on Apr 4, 2017

@author: ubuntu
'''
import tensorflow as tf

state = tf.Variable(0,name="counter")

value1 = tf.constant(1)

new_value = tf.add(state,value1)

product=tf.assign(state,new_value)

init_op = tf.initialize_all_variables()

with tf.Session() as sess:
    
    sess.run(init_op)
    print sess.run(state)
    
    for _ in range(3):
        sess.run(product)
        print sess.run(state)
'''
0
1
2
3
'''
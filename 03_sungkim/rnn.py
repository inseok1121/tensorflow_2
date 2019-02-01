import tensorflow as tf
import numpy as np

"""
shape=(a, b, c)

a : batch size
b : sequence_length : number of input data
c : dimension

"""

h = [1, 0, 0, 0]
e = [0, 1, 0, 0]
l = [0, 0, 1, 0]
o = [0, 0, 0, 1]

batch_size = 3
hidden_size = 2
sequence_length = 5

cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size)

x_data = np.array([[h, e, l, l, o],
                   [e, o, l, l, l],
                   [l, l, e, e, l]],
                  dtype=np.float32)

outputs, _states = tf.nn.dynamic_rnn(cell, x_data, dtype=tf.float32)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(outputs.eval())

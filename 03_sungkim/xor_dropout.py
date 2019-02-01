import tensorflow as tf
import numpy as np

x_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y_data = np.array([[0], [1], [1], [0]], dtype=np.float32)

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)
dropout_rate = tf.placeholder("float")

W1 = tf.Variable(tf.random_uniform([2, 5], -1.0, 1.0), name='weight1')
W2 = tf.Variable(tf.random_uniform([5, 5], -1.0, 1.0), name='weight2')
W3 = tf.Variable(tf.random_uniform([5, 5], -1.0, 1.0), name='weight3')
W4 = tf.Variable(tf.random_uniform([5, 5], -1.0, 1.0), name='weight4')
W5 = tf.Variable(tf.random_uniform([5, 5], -1.0, 1.0), name='weight5')
W6 = tf.Variable(tf.random_uniform([5, 5], -1.0, 1.0), name='weight6')
W7 = tf.Variable(tf.random_uniform([5, 5], -1.0, 1.0), name='weight7')
W8 = tf.Variable(tf.random_uniform([5, 5], -1.0, 1.0), name='weight8')
W9 = tf.Variable(tf.random_uniform([5, 5], -1.0, 1.0), name='weight9')
W10 = tf.Variable(tf.random_uniform([5, 1], -1.0, 1.0), name='weight10')

b1 = tf.Variable(tf.zeros([5]), name='bias1')
b2 = tf.Variable(tf.zeros([5]), name='bias2')
b3 = tf.Variable(tf.zeros([5]), name='bias3')
b4 = tf.Variable(tf.zeros([5]), name='bias4')
b5 = tf.Variable(tf.zeros([5]), name='bias5')
b6 = tf.Variable(tf.zeros([5]), name='bias6')
b7 = tf.Variable(tf.zeros([5]), name='bias7')
b8 = tf.Variable(tf.zeros([5]), name='bias8')
b9 = tf.Variable(tf.zeros([5]), name='bias9')
b10 = tf.Variable(tf.zeros([1]), name='bias10')

_layer1 = tf.nn.relu(tf.matmul(X, W1) + b1)
layer1 = tf.nn.dropout(_layer1, dropout_rate)
_layer2 = tf.nn.relu(tf.matmul(layer1, W2) + b2)
layer2 = tf.nn.dropout(_layer2, dropout_rate)
_layer3 = tf.nn.relu(tf.matmul(layer2, W3) + b3)
layer3 = tf.nn.dropout(_layer3, dropout_rate)
_layer4 = tf.nn.relu(tf.matmul(layer3, W4) + b4)
layer4 = tf.nn.dropout(_layer4, dropout_rate)
_layer5 = tf.nn.relu(tf.matmul(layer4, W5) + b5)
layer5 = tf.nn.dropout(_layer5, dropout_rate)
_layer6 = tf.nn.relu(tf.matmul(layer5, W6) + b6)
layer6 = tf.nn.dropout(_layer6, dropout_rate)
_layer7 = tf.nn.relu(tf.matmul(layer6, W7) + b7)
layer7 = tf.nn.dropout(_layer7, dropout_rate)
_layer8 = tf.nn.relu(tf.matmul(layer7, W8) + b8)
layer8 = tf.nn.dropout(_layer8, dropout_rate)
_layer9 = tf.nn.relu(tf.matmul(layer8, W9) + b9)
layer9 = tf.nn.dropout(_layer9, dropout_rate)

hypothesis = tf.sigmoid(tf.matmul(layer9, W10) + b10)

cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))
train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
acurracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(10001):
        sess.run(train, feed_dict={X: x_data, Y: y_data, dropout_rate: 1})


    h, c, a = sess.run([hypothesis, predicted, acurracy], feed_dict={X: x_data, Y: y_data, dropout_rate: 1})
    print("\nHypothesis : ", h, "\nCorrect: ", c, "\nAccuracy : ", a)

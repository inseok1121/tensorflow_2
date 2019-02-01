import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.layers import xavier_initializer


tf.set_random_seed(0)

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

test_x = mnist.test.images
test_y = mnist.test.labels


X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])
dropout_rate = tf.placeholder(tf.float32)

W1 = tf.get_variable("W1", shape=[784, 512], initializer=xavier_initializer())
W2 = tf.get_variable("W2", shape=[512, 256], initializer=xavier_initializer())
W3 = tf.get_variable("W3", shape=[256, 128], initializer=xavier_initializer())
W4 = tf.get_variable("W4", shape=[128, 64], initializer=xavier_initializer())
W5 = tf.get_variable("W5", shape=[64, 10], initializer=xavier_initializer())

b1 = tf.Variable(tf.random_normal([512]))
b2 = tf.Variable(tf.random_normal([256]))
b3 = tf.Variable(tf.random_normal([128]))
b4 = tf.Variable(tf.random_normal([64]))
b5 = tf.Variable(tf.random_normal([10]))

_layer1 = tf.nn.relu(tf.add(tf.matmul(X, W1), b1))
layer1 = tf.nn.dropout(_layer1, dropout_rate)
_layer2 = tf.nn.relu(tf.add(tf.matmul(layer1, W2), b2))
layer2 = tf.nn.dropout(_layer2, dropout_rate)
_layer3 = tf.nn.relu(tf.add(tf.matmul(layer2, W3), b3))
layer3 = tf.nn.dropout(_layer3, dropout_rate)
_layer4 = tf.nn.relu(tf.add(tf.matmul(layer3, W4), b4))
layer4 = tf.nn.dropout(_layer4, dropout_rate)

hypothesis = tf.add(tf.matmul(layer4, W5), b5)


cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(50):

        avg_cost = 0
        total_batch = int(mnist.train.num_examples / 100)

        for i in range(total_batch):

            train_x, train_y = mnist.train.next_batch(100)
            c, _ = sess.run([cost, optimizer], feed_dict={X: train_x, Y: train_y, dropout_rate:0.7})
            avg_cost += c / total_batch

        print("Epoch : ", epoch, "COST : ", avg_cost)

    print("Learning Finished\n")

    predict = tf.equal(tf.argmax(Y, 1), tf.argmax(hypothesis, 1))
    accuracy = tf.reduce_mean(tf.cast(predict, tf.float32))

    print("\nAccuracy : ", sess.run(accuracy, feed_dict={X: test_x, Y: test_y, dropout_rate: 1}))


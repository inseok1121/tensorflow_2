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


W1 = tf.Variable(tf.random_normal([784, 256]))
W5 = tf.Variable(tf.random_normal([256, 10]))

b1 = tf.Variable(tf.random_normal([256]))
b5 = tf.Variable(tf.random_normal([10]))

layer1 = tf.nn.relu(tf.add(tf.matmul(X, W1), b1))
hypothesis = tf.add(tf.matmul(layer1, W5), b5)


cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(50):

        avg_cost = 0
        total_batch = int(mnist.train.num_examples / 100)

        for i in range(total_batch):

            train_x, train_y = mnist.train.next_batch(100)
            c, _ = sess.run([cost, optimizer], feed_dict={X: train_x, Y: train_y})
            avg_cost += c / total_batch

        print("Epoch : ", epoch, "COST : ", avg_cost)

    print("Learning Finished\n")

    predict = tf.equal(tf.argmax(Y, 1), tf.argmax(hypothesis, 1))
    accuracy = tf.reduce_mean(tf.cast(predict, tf.float32))

    print("\nAccuracy : ", sess.run(accuracy, feed_dict={X: test_x, Y: test_y}))


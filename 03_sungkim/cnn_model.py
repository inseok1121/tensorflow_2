import tensorflow as tf

import tensorflow.contrib.layers as ad

tf.set_random_seed(0)
learning_rate = 0.001
training_epochs = 15
batch_size = 100

class Model:

    def __init__(self, sess, name):
        self.sess = sess
        self.name = name
        self._build_net()

    def _build_net(self):
        with tf.variable_scope(self.name):
            self.X = tf.placeholder(tf.float32, [None, 784])
            self.Y = tf.placeholder(tf.float32, [None, 10])
            self.keep_prob = tf.placeholder(tf.float32)

            x_img = tf.reshape(self.X, [-1, 28, 28, 1])

            W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))

            L1 = tf.nn.conv2d(x_img, W1, strides=[1, 1, 1, 1], padding='SAME')
            L1 = tf.nn.relu(L1)
            L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

            # 2 convolution
            W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))

            L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
            L2 = tf.nn.relu(L2)
            L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            L2 = tf.reshape(L2, [-1, 7 * 7 * 64])

            # Fully Connected
            W3 = tf.get_variable("W2", shape=[7 * 7 * 64, 10], initializer=ad.xavier_initializer())
            b = tf.Variable(tf.random_normal([10]))
            self.logits = tf.matmul(L2, W3) + b

            self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y))
            self.optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.cost)

            predict = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.Y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(predict, tf.float32))

    def predict(self, x_test, keep_prop=1.0):
        return self.sess.run(self.logits, feed_dict={self.X: x_test, self.keep_prob: keep_prop})

    def get_accuracy(self, x_test, y_test, keep_prop=1.0):
        return self.sess.run(self.accuracy, feed_dict={self.X: x_test, self.Y: y_test, self.keep_prob: keep_prop})

    def train(self, x_data, y_data, keep_prop=0.7):
        return self.sess.run([self.cost, self.optimizer], feed_dict={self.X: x_data, self.Y: y_data, self.keep_prob: keep_prop})
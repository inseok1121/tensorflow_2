import tensorflow as tf
import numpy as np

x_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y_data = np.array([[0], [1], [1], [0]], dtype=np.float32)

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W1= tf.Variable(tf.random_normal([2, 16]), name='weight1')
b1 = tf.Variable(tf.random_normal([16]), name='bias1')
layer1 = tf.sigmoid(tf.matmul(X, W1) + b1)

W2= tf.Variable(tf.random_normal([16, 8]), name='weight2')
b2 = tf.Variable(tf.random_normal([8]), name='bias2')
layer2 = tf.sigmoid(tf.matmul(layer1, W2) + b2)

W3= tf.Variable(tf.random_normal([8, 4]), name='weight3')
b3 = tf.Variable(tf.random_normal([4]), name='bias3')
layer3 = tf.sigmoid(tf.matmul(layer2, W3) + b3)

W4= tf.Variable(tf.random_normal([4, 2]), name='weight4')
b4 = tf.Variable(tf.random_normal([2]), name='bias4')
layer4 = tf.sigmoid(tf.matmul(layer3, W4) + b4)

W5 = tf.Variable(tf.random_normal([2, 1]), name='weight5')
b5 = tf.Variable(tf.random_normal([1]), name='bias5')
hypothesis = tf.sigmoid(tf.matmul(layer4, W5) + b5)

cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))
train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
acurracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(10001):
        sess.run(train, feed_dict={X: x_data, Y:y_data})


    h, c, a = sess.run([hypothesis, predicted, acurracy], feed_dict={X: x_data, Y: y_data})
    print("\nHypothesis : ", h, "\nCorrect: ", c, "\nAccuracy : ", a)

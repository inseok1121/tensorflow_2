import tensorflow as tf

x_data = [[2., 0.], [4., 4.], [6., 2.], [8., 3.]]
y_data = [[81.], [93.], [91.], [97.]]
X = tf.placeholder(dtype=tf.float32, shape=[None, 2])
Y = tf.placeholder(dtype=tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([2, 1]))
b = tf.Variable(tf.random_normal([1]))

y = tf.matmul(X, W) + b

cost = tf.sqrt(tf.reduce_mean(tf.square(y - Y)))
learning_rate = 0.1

train = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(2001):
        sess.run(train, feed_dict={X:x_data, Y:y_data})
        if step % 100 == 0:
            print("%.f, cost = %.04f, W = %.4f,  b = %.4f" % (step, sess.run(cost, feed_dict={X:x_data, Y:y_data}), sess.run(W[0], feed_dict={X:x_data, Y:y_data}), sess.run(b, feed_dict={X:x_data, Y:y_data})))

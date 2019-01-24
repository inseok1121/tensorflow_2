import tensorflow as tf

x_data = [2., 4., 6., 8., 10., 12., 14.]
y_data = [0., 0., 0., 1., 1., 1., 1.]

X = tf.placeholder(dtype=tf.float64, shape=[None])
Y = tf.placeholder(dtype=tf.float64, shape=[None])

a = tf.Variable(tf.random_normal([1], dtype=tf.float64, seed=0))
b = tf.Variable(tf.random_normal([1], dtype=tf.float64, seed=0))

y = tf.sigmoid(a*X+b)

cost = -tf.reduce_mean(Y * tf.log(y) + (1 - Y) * tf.log(1 - y))
learning_rate = 0.5

train = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(60001):
        sess.run(train, feed_dict={X: x_data, Y: y_data})
        if step % 6000 == 0:
            print("%.f, cost = %.04f, W = %.4f,  b = %.4f" % (step, sess.run(cost, feed_dict={X:x_data, Y:y_data}), sess.run(a, feed_dict={X:x_data, Y:y_data}), sess.run(b, feed_dict={X:x_data, Y:y_data})))

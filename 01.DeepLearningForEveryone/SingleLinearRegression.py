import tensorflow as tf


x_data = [2., 4., 6., 8.]
y_data = [81., 93., 91., 97.]

X = tf.placeholder(dtype=tf.float32, shape=[None])
Y = tf.placeholder(dtype=tf.float32, shape=[None])

a = tf.Variable(tf.random.uniform([1], 0, 10, dtype=tf.float32, seed=0))
b = tf.Variable(tf.random.uniform([1], 0, 100, dtype=tf.float32, seed=0))

y = a * x_data + b

cost = tf.sqrt(tf.reduce_mean(tf.square(y - y_data)))
learning_rate = 0.1

train = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(20001):
        sess.run(train, feed_dict={X:x_data, Y:y_data})
        if step % 100 == 0:
            print("%.f, cost = %.04f, LEAN = %.4f, b = %.4f" % (step, sess.run(cost), sess  .run(a), sess.run(b)))
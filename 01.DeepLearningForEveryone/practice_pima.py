import tensorflow as tf
import numpy as np

seed = 0

tf.set_random_seed(seed)

dataset = np.loadtxt('./dataset/dataset_pimaindian.csv', delimiter=',', dtype=np.float32)

x_data = dataset[:, 0:-1]
y_data = dataset[:, [0-1]]


X = tf.placeholder(dtype=tf.float32, shape=[None, 8])
Y = tf.placeholder(dtype=tf.float32, shape=[None, 1])

n_layer1 = 12
n_layer2 = 8
n_layer3 = 1

learning_rate = 0.001

w = {
    'h1': tf.Variable(tf.random_normal([8, n_layer1], mean=0.01, stddev=0.01)),
    'h2': tf.Variable(tf.random_normal([n_layer1, n_layer2], mean=0.01, stddev=0.01)),
    'h3': tf.Variable(tf.random_normal([n_layer2, n_layer3], mean=0.01, stddev=0.01)),
    'out': tf.Variable(tf.random_normal([n_layer3, 1], mean=0.01, stddev=0.01))
}
b = {
    'b1': tf.Variable(tf.random_uniform([n_layer1])),
    'b2': tf.Variable(tf.random_uniform([n_layer2])),
    'b3': tf.Variable(tf.random_uniform([n_layer3])),
    'out': tf.Variable(tf.random_uniform([1]))
}

layer_1 = tf.add(tf.matmul(X, w['h1']), b['b1'])
layer_1 = tf.nn.relu(layer_1)
layer_2 = tf.add(tf.matmul(layer_1, w['h2']), b['b2'])
layer_2 = tf.nn.relu(layer_2)
layer_3 = tf.add(tf.matmul(layer_2, w['h3']), b['b3'])
layer_4 = tf.nn.relu(layer_3)

out_layer = tf.add(tf.matmul(layer_3, w['out']), b['out'])

pred = out_layer

cost = -tf.reduce_mean(Y * tf.log(tf.sigmoid(out_layer)) + (1 - Y) * tf.log(1 - tf.sigmoid(out_layer)))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)


init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(20001):

        result = sess.run([optimizer, cost], feed_dict={X: x_data, Y: y_data})
        if epoch % 1000 == 0:
            print("epoch : %d cost : %.4f"%(epoch, result[1]))

    correct_prediction = tf.cast(tf.sigmoid(out_layer) > 0.5, dtype=tf.float32)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(correct_prediction, Y), tf.float32))
    print("Accuracy : %.4f"%(sess.run(accuracy, feed_dict={X: x_data, Y: y_data})))

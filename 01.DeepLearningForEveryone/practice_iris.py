import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

seed = 0
tf.set_random_seed(seed)

df = pd.read_csv('./dataset/dataset_iris.csv', names=["sepal_length", "sepal_width", "petal_length", "petal_width", "species"])
dataset = df.values

x_data = dataset[:, 0:-1].astype(float)

y_data_obj = dataset[:, [-1]]
e = LabelEncoder()
e.fit(y_data_obj)
y_data = e.transform(y_data_obj)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

y_data = tf.one_hot(y_data, depth=3).eval(session=sess)


X = tf.placeholder(dtype=tf.float32, shape=[None, 4])
Y = tf.placeholder(dtype=tf.float32, shape=[None, 3])

w = {
    'h1': tf.Variable(tf.random_normal([4, 16], mean=0.01, stddev=0.01)),
    'out': tf.Variable(tf.random_normal([16, 3], mean=0.01, stddev=0.01))
}
b = {
    'b1': tf.Variable(tf.random_uniform([16])),
    'out': tf.Variable(tf.random_uniform([3]))
}

layer_1 = tf.add(tf.matmul(X, w['h1']), b['b1'])
layer_1 = tf.nn.relu(layer_1)

layer_out = tf.add(tf.matmul(layer_1, w['out']), b['out'])
layer_out = tf.nn.softmax(layer_out)

learning_rate = 0.001

cost = -tf.reduce_mean(Y * tf.log(layer_out) + (1 - Y) * tf.log(1 - layer_out))
train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

with tf.Session() as sess:

    #merged_summary = tf.summary.merge_all()
    #writer = tf.summary.FileWriter("./logs/pima_tf-1")
    #writer.add_graph(sess.graph)
    sess.run(init)
    for epoch in range(2001):

        result = sess.run([train, cost], feed_dict={X: x_data, Y: y_data})
        if epoch % 1000 == 0:
            print("epoch : %d cost : %.4f"%(epoch, result[1]))

    """
    correct_prediction = tf.cast(layer_out > 0.5, dtype=tf.float32)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(correct_prediction, Y), tf.float32))
    print("Accuracy : %.4f"%(sess.run(accuracy, feed_dict={X: x_data, Y: y_data})))
    """
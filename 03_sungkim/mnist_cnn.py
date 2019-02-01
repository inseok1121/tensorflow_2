import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt


tf.set_random_seed(0)

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

image = mnist.train.images[0]
img = image.reshape(-1, 28, 28, 1)

sess = tf.InteractiveSession()

W1 = tf.Variable(tf.random_normal([3, 3, 1, 5], stddev=0.01))
conv2d = tf.nn.conv2d(img, W1, strides=[1, 2, 2, 1], padding='SAME')


pool = tf.nn.max_pool(conv2d, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
sess.run(tf.global_variables_initializer())

pool_img = pool.eval()

pool_img = np.swapaxes(pool_img, 0, 3)

for i, one_img in enumerate(pool_img):
    plt.subplot(1, 5, i+1), plt.imshow(one_img.reshape(7, 7), cmap='gray')

plt.show()
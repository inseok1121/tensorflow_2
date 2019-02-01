from cnn_model import Model
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
sess = tf.Session()

models = []
num_models = 7

for m in range(num_models):
    models.append(Model(sess, "model"+str(m)))


sess.run(tf.global_variables_initializer())
print("Learning Started\n")

for epoch in range(15):
    avg_cost_list = np.zeros(len(models))
    total_batch = int(mnist.train.num_examples / 100)

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(100)

        for m_idx, m  in enumerate(models):
            c, _ = m.train(batch_xs, batch_ys)
            avg_cost_list[m_idx] += c / total_batch

    print("Epoch : ", epoch, "Cost : " , avg_cost_list)

print('Learning Finished\n')


test_size = len(mnist.test.labels)
predictions = np.zeros(test_size * 10).reshape(test_size, 10)

for m_idx, m in enumerate(models):
    print(m_idx, 'Accuracy : ', m.get_accuracy(mnist.test.images, mnist.test.labels))
    p = m.predict(mnist.test.images)
    predictions += p

ensemble_correct_prediction = tf.equal(tf.argmax(predictions, 1), tf.argmax(mnist.test.labels, 1))
ensemble_accuracy = tf.reduce_mean(tf.cast(ensemble_correct_prediction, tf.float32))
print('Ensemble Accuracy : ', sess.run(ensemble_accuracy))




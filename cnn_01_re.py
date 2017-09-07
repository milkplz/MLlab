# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

tf.set_random_seed(777)  # reproducibility

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# Check out https://www.tensorflow.org/get_started/mnist/beginners for
# more information about the mnist dataset

# hyper parameters
learning_rate = 0.001
# training_epochs = 15
training_epochs = 1
batch_size = 100



X = tf.placeholder(tf.float32, [None, 784])
# img 28x28x1 
X_img = tf.reshape(X, [-1, 28, 28, 1])
Y = tf.placeholder(tf.float32, [None, 10])

# L1 imgIn shape=(?, 28, 28, 1)
# W -> random로 초기화
W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01), name="W1")
# Conv -> (?, 28, 28, 32)
# Pool -> (?, 14, 14, 32)
L1 = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1], padding='SAME')
L1 = tf.nn.relu(L1)
L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# L2 imgIn shape=(?, 14, 14, 32)
# W -> random로 초기화
W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01), name="W2")
# Conv -> (?, 14, 14, 64)
# Pool -> (?, 7, 7, 64)
L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
# Fully connected layer에 넣기 위해 1줄로 reshape
L2 = tf.reshape(L2, [-1, 7 * 7 * 64])

# Final FC 7x7x64 inputs -> 10 outputs
# W -> xavier로 초기화
W3 = tf.get_variable("W3", shape=[7 * 7 * 64, 10], initializer=tf.contrib.layers.xavier_initializer())
b = tf.Variable(tf.random_normal([10]), name="b")
hypothesis = tf.matmul(L2, W3) + b

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)



saver = tf.train.Saver()


# session open
sess = tf.Session()
# sess.run(tf.global_variables_initializer())

saver.restore(sess, "./model/cnn.ckpt")


print('Learning Start')
for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size)
        for i in range(total_batch):
                batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                feed_dict = {X: batch_xs, Y: batch_ys}
                c, _, = sess.run([cost, optimizer], feed_dict=feed_dict)
                avg_cost += c / total_batch
        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

print('Learning Finished!')

# Test model and check accuracy
correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print('Accuray:', sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels}))

save_path = saver.save(sess, "./model/cnn.ckpt")
print "Model saved in file: ", save_path
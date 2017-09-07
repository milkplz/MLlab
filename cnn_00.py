import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
minst = input_data.read_data_sets("MNIST_data/", one_hot=True)

img = minst.train.images[0].reshape(28,28)

sess = tf.InteractiveSession()

img = img.reshape(-1,28,28,1)
# reshape(image num, width, height, color channel)
Wl = tf.Variable(tf.random_normal([3, 3, 1, 5], stddev=0.01))
# Wl은 filter를 의미. w를 randown하게 생성 -> tf.random_normal([width, heioght, channel, filter num], stddev=0.01)
# filter num에 따라 출력되는 이미지의 갯수가 결정됨.
conv2d = tf.nn.conv2d(img, Wl, strides=[1,2,2,1], padding='SAME')
# strides=[1,filter를 가로로 움직이는 거리, filter를 세로로 움직이는 거리, 1]
# strides -> filter가 움직이는 거리를 정의
print(conv2d)

# sess.run(tf.global_variables_initializer())
# conv2d_img = conv2d.eval()
# conv2d.eval() -> 이미지를 출력 시키키 위한 함수
# conv2d_img = np.swapaxes(conv2d_img, 0, 3)
# for i, one_img in enumerate(conv2d_img):
    # plt.subplot(1,5,i+1), plt.imshow(one_img.reshape(14,14), cmap='gray')

pool = tf.nn.max_pool(conv2d, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
# ksize=[1,width,height,1]
print(pool)
sess.run(tf.global_variables_initializer())
pool_img = pool.eval()
pool_img = np.swapaxes(pool_img, 0, 3)
for i, one_img in enumerate(pool_img):
    plt.subplot(1,5,i+1), plt.imshow(one_img.reshape(7,7), cmap='gray')

plt.show()


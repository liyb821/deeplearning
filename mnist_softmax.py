# coding:utf8

import sys
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#print(mnist.train.images.shape, mnist.train.labels.shape)
#print(mnist.test.images.shape, mnist.test.labels.shape)
#print(mnist.validation.images.shape, mnist.validation.labels.shape)

# step 1: build the net architecture
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])
w = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
# softmax
y_ = tf.nn.softmax(tf.matmul(x, w) + b)

# step 2: define the loss and the optimizer
# loss function
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y*tf.log(y_), reduction_indices=[1]))

# optimizer
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# step 3: train
# init
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

for i in range(1000):
    batch_x, batch_y = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x:batch_x, y:batch_y})

# step 4: eval
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print sess.run(accuracy, {x:mnist.test.images, y:mnist.test.labels})

sess.close()



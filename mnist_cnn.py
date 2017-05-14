# coding:utf8

import sys
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

def bias_variable(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))

def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding="SAME")

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


# step1: net architecture
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])
x_image = tf.reshape(x, [-1, 28, 28, 1])

w1 = weight_variable([5, 5 ,1, 32])
b1 = bias_variable([32])

h_conv1 = tf.nn.relu(conv2d(x_image, w1) + b1)
h_pool1 = max_pool_2x2(h_conv1)

w2 = weight_variable([5, 5 ,32, 64])
b2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, w2) + b2)
h_pool2 = max_pool_2x2(h_conv2)

w3 = weight_variable([7*7*64, 1024])
b3 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w3) + b3)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

w4 = weight_variable([1024, 10])
b4 = bias_variable([10])
y_ = tf.nn.softmax(tf.matmul(h_fc1_drop, w4) + b4)

# step2: loss and optimizer
# loss function
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y*tf.log(tf.clip_by_value(y_,1e-10,1.0)), reduction_indices=[1]))
# optimizer
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# step3: train
# init
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
for i in range(10000):
    batch_x, batch_y = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x:batch_x, y:batch_y, keep_prob:0.75})

    if i % 100 == 0:
        print "epoch %d, the loss is %f" % (i, sess.run(cross_entropy, {x:batch_x, y:batch_y, keep_prob:1.0}))

# step4: eval
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print sess.run(accuracy, {x:mnist.test.images, y:mnist.test.labels, keep_prob:1.0})

sess.close()



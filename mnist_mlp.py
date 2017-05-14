# coding:utf8

import sys
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#print(mnist.train.images.shape, mnist.train.labels.shape)
#print(mnist.test.images.shape, mnist.test.labels.shape)
#print(mnist.validation.images.shape, mnist.validation.labels.shape)

h1_units = 300
h2_units = 200
keep_prob = tf.placeholder(tf.float32)

# step1: net architecture
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])
w1 = tf.Variable(tf.truncated_normal([784, h1_units], stddev=0.1))
b1 = tf.Variable(tf.zeros([h1_units]))
w2 = tf.Variable(tf.truncated_normal([h1_units, h2_units], stddev=0.1))
b2 = tf.Variable(tf.zeros([h2_units]))
w3 = tf.Variable(tf.zeros([h2_units, 10]))
b3 = tf.Variable(tf.zeros([10]))
hidden1 = tf.nn.relu(tf.matmul(x, w1) + b1)
hidden1_drop = tf.nn.dropout(hidden1, keep_prob)
hidden2 = tf.nn.relu(tf.matmul(hidden1_drop, w2) + b2)
hidden2_drop = tf.nn.dropout(hidden2, keep_prob)
y_ = tf.nn.softmax(tf.matmul(hidden2_drop, w3) + b3)

# step2: loss and optimizer
# loss function
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y*tf.log(tf.clip_by_value(y_,1e-10,1.0)), reduction_indices=[1]))
all_weights = tf.trainable_variables()
l2_loss = tf.constant(0.)
for each_weight in all_weights:
    l2_loss += tf.nn.l2_loss(each_weight) / tf.cast(tf.size(each_weight), tf.float32)
total_loss = cross_entropy + 0.1*l2_loss
# optimizer
#train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
train_step = tf.train.AdagradOptimizer(0.3).minimize(total_loss)

# step3: train
# init
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
print sess.run(l2_loss)
for i in range(10000):
    batch_x, batch_y = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x:batch_x, y:batch_y, keep_prob:0.75})

    if i % 100 == 0:
        print "epoch %d, the loss is %f" % (i, sess.run(total_loss, {x:batch_x, y:batch_y, keep_prob:1.0}))

# step4: eval
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print sess.run(accuracy, {x:mnist.test.images, y:mnist.test.labels, keep_prob:1.0})

sess.close()



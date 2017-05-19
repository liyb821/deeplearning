# coding:utf8

import sys
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./tmp/data/", one_hot=True)

# parameter
n_input = 784
n_hidden1 = 256
n_hidden2 = 128
batch_size = 256
training_epochs = 100

weights = {
    'input_h1':tf.Variable(tf.random_normal([n_input, n_hidden1])),
    'input_h2':tf.Variable(tf.random_normal([n_hidden1, n_hidden2])),
    'output_h1':tf.Variable(tf.random_normal([n_hidden2, n_hidden1])),
    'output_h2':tf.Variable(tf.random_normal([n_hidden1, n_input])),
}

biases = {
    'input_h1':tf.Variable(tf.random_normal([n_hidden1])),
    'input_h2':tf.Variable(tf.random_normal([n_hidden2])),
    'output_h1':tf.Variable(tf.random_normal([n_hidden1])),
    'output_h2':tf.Variable(tf.random_normal([n_input])),
}

# basic function
def encoder(x, weights, biases):
    hidden1 = tf.nn.sigmoid(tf.matmul(x, weights['input_h1']) + biases['input_h1'])
    hidden2 = tf.nn.sigmoid(tf.matmul(hidden1, weights['input_h2']) + biases['input_h2'])
    return hidden2

def decoder(x, weights, biases):
    hidden1 = tf.nn.sigmoid(tf.matmul(x, weights['output_h1']) + biases['output_h1'])
    hidden2 = tf.nn.sigmoid(tf.matmul(hidden1, weights['output_h2']) + biases['output_h2'])
    return hidden2

# build net architecture
x = tf.placeholder(tf.float32, [None, n_input], name='x')

code = encoder(x, weights, biases)
y_ = decoder(code, weights, biases)

# define the loss function and the optimizer
loss = tf.reduce_mean(tf.pow(y_-x, 2))
#optimizer = tf.train.AdagradOptimizer(0.3).minimize(loss)
optimizer = tf.train.RMSPropOptimizer(0.01).minimize(loss)

# train
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    total_batch = int(mnist.train.num_examples/batch_size)
    for epoch in range(training_epochs):
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            _, l = sess.run([optimizer, loss], feed_dict={x:batch_xs})

        if epoch % 1 == 0:
            print "%d epoch, loss is %f" % (epoch, l)


# eval





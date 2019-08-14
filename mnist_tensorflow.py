# -*- coding: utf-8 -*-
"""
Created on Tue Jan  1 16:49:45 2019

@author: elena
"""

import tensorflow as tf
#mnist = tf.keras.datasets.mnist
from tensorflow.examples.tutorials.mnist import input_data

#1. Load MNIST data
mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)

#2. Create a placeholder with the shape: [ None , 784 ]
x = tf.placeholder("float", [None, 784]) # MNIST data image of shape 28*28=784

#3. Create W variable with the shaep: [ 784 , 10 ]
W = tf.Variable(tf.zeros([784, 10]))

#4. Create b variable with the shape: [ 10 ]
b = tf.Variable(tf.zeros([10]))

#5. The neural network layer will be represented by, we will learn more in the future:
model = tf.nn.softmax(tf.matmul(x, W) + b)

#6. Create a placeholder y (labels) with the shape: [ None , 10 ]
y = tf.placeholder("float", [None, 10]) # 0-9 digits recognition => 10 classes


#7. Our loss funcction will be the cross entropy:
#We will use reduce_sum and then reduce_mean
loss_function = -tf.reduce_sum(y*tf.log(model))

#8. We will build a gradient descent optimizer and then use the method minimize to get a training_step
learning_rate = 0.01
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_function)

#9. Run this optimizer for 1000 iterations, each time on the next batch of data. The function
#mnist.train.next_batch( 100 ) can be used to get the next batch_xs, batch_ys
#of the data, and then these values can be fed to the feed_dict in the run function
training_iteration = 100
batch_size = 100
display_step = 5
total_batch = int(mnist.train.num_examples/batch_size)
init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    for iteration in range(training_iteration):
        avg_cost = 0.
        for i in range(total_batch):       
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})
            avg_cost += sess.run(loss_function, feed_dict={x: batch_xs, y: batch_ys})/total_batch
        if iteration % display_step == 0:
            print ("Iteration:", '%04d' % (iteration + 1), "cost=", "{:.9f}".format(avg_cost))
    print ("Tuning completed!")

#10. Run the model trained on the test data which can be accesses using x: mnist.test.images, y_: mnist.test.labels
#11. Count the number of correct predictions
    correct_predictions = tf.equal(tf.argmax(model, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"))
    print ("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))


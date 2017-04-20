"""
Starter code for logistic regression model to solve OCR task
with MNIST in TensorFlow
MNIST dataset: yann.lecun.com/exdb/mnist/
"""

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import time

# Define paramaters for the model
learning_rate = 0.01
batch_size = 128
n_epochs = 40
log_step = 2

# Step 1: Read in data
# using TF Learn's built in function to load MNIST data to the folder data/mnist
mnist = input_data.read_data_sets('data/mnist', one_hot=True)

with tf.name_scope("Input"):
    # Step 2: create placeholders for features and labels
    # each image in the MNIST data is of shape 28*28 = 784
    # therefore, each image is represented with a 1x784 tensor
    # there are 10 classes for each image, corresponding to digits 0 - 9. 
    X = tf.placeholder(tf.float32, [None, 784], name="image")
    Y = tf.placeholder(tf.float32, [None, 10], name="label")

# Step 3: create weights and bias
# weights and biases are initialized to 0
# shape of w depends on the dimension of X and Y so that Y = X * w + b
# shape of b depends on Y
with tf.name_scope("weights"):
    w = tf.Variable(tf.truncated_normal([784, 10], stddev=0.1), name='w')

with tf.name_scope("biases"):
    b = tf.Variable(tf.constant(0.1, shape=[1, 10]), name='b')

# Step 4: build model
# the model that returns the logits.
# this logits will be later passed through softmax layer
# to get the probability distribution of possible label of the image
# DO NOT DO SOFTMAX HERE
with tf.name_scope("Model"):
        logits = tf.add(tf.matmul(X, w), b, name='logits')

with tf.name_scope("Loss"):
    # Step 5: define loss function
    # use cross entropy loss of the real labels with the softmax of logits
    # use the method:
    # tf.nn.softmax_cross_entropy_with_logits(logits, Y)
    # then use tf.reduce_mean to get the mean loss of the batch
    entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y, name="entropy")
    loss = tf.reduce_mean(entropy, name='loss')

# Step 6: define training op
# using gradient descent to minimize loss
with tf.name_scope("SDG"):
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# Accuracy
with tf.name_scope("Evaluation"):
    Y_hat = tf.nn.softmax(logits)
    correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_hat, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Create operation to initialize all variables
init = tf.global_variables_initializer()

# Create a summary to monitor loss
tf.summary.scalar("accuracy", accuracy, collections=['train', 'test'])
tf.summary.scalar("loss", loss, collections=['train', 'test'])

# merge summaries per collection
training_summary = tf.summary.merge_all('train')
validation_summary = tf.summary.merge_all('test')

with tf.Session() as sess:
    writer_test = tf.summary.FileWriter('./graphs/logistic/train', sess.graph)
    writer_val = tf.summary.FileWriter('./graphs/logistic/val', sess.graph)

    start_time = time.time()
    sess.run(init)

    # train the model n_epochs times
    n_batches = int(mnist.train.num_examples/batch_size)

    for i in range(n_epochs):
        epoch_loss = 0.0
        for _ in range(n_batches):
            # fetch new batch
            X_batch, Y_batch = mnist.train.next_batch(batch_size)

            # run ops
            _, loss_batch = sess.run([optimizer, loss],
                                     feed_dict={X: X_batch, Y: Y_batch})
            # update loss
            epoch_loss += loss_batch

        print('Average loss epoch {0}: {1}'.format(i, epoch_loss/n_batches))

        if i % log_step == 0:
            # To log training accuracy.
            train_acc, train_summ = sess.run([accuracy, training_summary],
                                             feed_dict={
                                                 X: mnist.train.images,
                                                 Y: mnist.train.labels})
            print('\tTraining accuracy {0}: {1}'.format(i, train_acc))
            writer_test.add_summary(train_summ, i)

            # To log validation accuracy.
            valid_acc, valid_summ = sess.run([accuracy, validation_summary],
                                             feed_dict={
                                                 X: mnist.validation.images,
                                                 Y: mnist.validation.labels})
            print('\tValidation accuracy {0}: {1}'.format(i, valid_acc))
            writer_val.add_summary(valid_summ, i)

    print('Total time: {0} seconds'.format(time.time() - start_time))
    print('Optimization Finished!')

    # test the model
    test_acc = sess.run(accuracy, feed_dict={
        X: mnist.test.images,
        Y: mnist.test.labels})

    print('Accuracy {0}'.format(test_acc))

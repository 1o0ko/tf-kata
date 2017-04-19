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
n_epochs = 20

# Step 1: Read in data
# using TF Learn's built in function to load MNIST data to the folder data/mnist
mnist = input_data.read_data_sets('data/mnist', one_hot=True)

# Step 2: create placeholders for features and labels
# each image in the MNIST data is of shape 28*28 = 784
# therefore, each image is represented with a 1x784 tensor
# there are 10 classes for each image, corresponding to digits 0 - 9. 
X = tf.placeholder(tf.float32, [batch_size, 784], name="image")
Y = tf.placeholder(tf.float32, [batch_size, 10], name="label")

# Step 3: create weights and bias
# weights and biases are initialized to 0
# shape of w depends on the dimension of X and Y so that Y = X * w + b
# shape of b depends on Y
w = tf.Variable(tf.zeros([784, 10]), name='w')
b = tf.Variable(tf.zeros([1, 10]), name='b')

# Step 4: build model
# the model that returns the logits.
# this logits will be later passed through softmax layer
# to get the probability distribution of possible label of the image
# DO NOT DO SOFTMAX HERE

# logits = X * w + b
logits = tf.add(tf.matmul(X, w), b, name='logits')

# Step 5: define loss function
# use cross entropy loss of the real labels with the softmax of logits
# use the method:
# tf.nn.softmax_cross_entropy_with_logits(logits, Y)
# then use tf.reduce_mean to get the mean loss of the batch
entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y)
loss = tf.reduce_mean(entropy, name='loss')

# Step 6: define training op
# using gradient descent to minimize loss
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# Create operation to initialize all variables
init = tf.global_variables_initializer()

# Create a summary to monitor loss
training_summary = tf.summary.scalar("loss", loss)

with tf.Session() as sess:
    writer = tf.summary.FileWriter('./graphs/logistic', sess.graph)

    start_time = time.time()
    sess.run(init)

    # train the model n_epochs times
    n_batches = int(mnist.train.num_examples/batch_size)


    prev_epoch_loss = 0.0
    for i in range(n_epochs):
        epoch_loss = 0.0
        for _ in range(n_batches):
            X_batch, Y_batch = mnist.train.next_batch(batch_size)
            _, loss_batch, summary = sess.run(
                [optimizer, loss, training_summary],
                feed_dict={X: X_batch, Y: Y_batch}
            )

            epoch_loss += loss_batch

        # Allow the training to quit if we've reached a minimum
        if np.abs(prev_epoch_loss - epoch_loss) < 0.00001:
            print("Eh, there's no point in trying")
            break
        prev_epoch_loss = epoch_loss

        print('Average loss epoch {0}: {1}'.format(i, epoch_loss/n_batches))

    print('Total time: {0} seconds'.format(time.time() - start_time))

    print('Optimization Finished!') # should be around 0.35 after 25 epochs

    # test the model
    n_batches = int(mnist.test.num_examples/batch_size)
    total_correct_preds = 0
    for i in range(n_batches):
        X_batch, Y_batch = mnist.test.next_batch(batch_size)
        _, loss_batch, logits_batch = sess.run([optimizer, loss, logits], feed_dict={X: X_batch, Y:Y_batch})
        preds = tf.nn.softmax(logits_batch)
        correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(Y_batch, 1))
        accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))
        total_correct_preds += sess.run(accuracy)

    print('Accuracy {0}'.format(total_correct_preds/mnist.test.num_examples))

    writer.close()

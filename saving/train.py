"""
Usage: train.py PATH

Arguments:
    PATH    path to save model
"""
import os

import numpy as np
import tensorflow as tf

from docopt import docopt

def gen_data(n, a=3.0, b=1.0):
   x = np.linspace(-5, 5, n)
   epsilon = np.random.normal(0, 0.1, n)
   y = a*x + b + epsilon

   return x.reshape(n, 1), y.reshape(n, 1)

def main(args):
    # Phase 1: Assemble the graph

    # Step 1: read in data from the .xls file
    theta  = (3, 1)
    x_data, y_data = gen_data(100, *theta)

    # Step 2: create placeholders for input X (number of fire) and label Y (number of theft)
    X = tf.placeholder(tf.float32, shape=[None, 1], name = "X")
    Y = tf.placeholder(tf.float32, shape=[None, 1], name = "Y")

    # Step 3: create weight and bias, initialized to 0
    w = tf.Variable(tf.zeros([1,1]), name='w')
    b = tf.Variable(tf.zeros([1,1]), name='b')

    # Step 4: predict Y (number of theft) from the number of fire
    with tf.name_scope("Model"):
        Y_predicted = tf.add(tf.matmul(X, w), b, name='Y_predicted')

    # Step 5: use the square error as the loss function
    with tf.name_scope("Loss"):
        loss = tf.reduce_mean(tf.square(Y - Y_predicted, name = 'loss'))

    # Step 6: using gradient descent with learning rate of 0.01 to minimize loss
    with tf.name_scope("SDG"):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)

    # Create operation to initialize all variables
    init = tf.global_variables_initializer()

    # Create a saver
    saver = tf.train.Saver()

    # Phase 2: Train our model
    with tf.Session() as sess:
        # Step 7: initialize the necessary variables, in this case, w and b
        sess.run(init)

        # Step 8: train the model
        for i in range(500): # run 500 epochs
            # Session runs optimizer to minimize loss and fetch the value of loss
            _, loss_, = sess.run([optimizer, loss],
                                 feed_dict={X: x_data, Y: y_data})

            if i % 10 == 0:
                print("Epoch {0}: {1}".format(i, loss_))

        # Step 9: output the values of w and b
        w_value, b_value = sess.run([w, b])
        print("w_hat: %0.2f, b_hat: %0.2f" % (w_value, b_value))
        print("w: %0.2f, b: %0.2f" % theta)

        saver.save(sess, os.path.join(args['PATH'], 'model'))



if __name__ == '__main__':
    args = docopt(__doc__)
    main(args)

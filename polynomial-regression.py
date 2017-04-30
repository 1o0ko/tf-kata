"""Simple tutorial for using TensorFlow to compute polynomial regression.
Parag K. Mital, Jan. 2016"""
# %% Imports
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from data import load_fire_theft

DATA_FILE = 'data/fire_theft.xls'
LAMBDA = 1e-06
POL_DEG = 5
learning_rate = 0.01
n_epochs = 500


def main():

    # Phase 1: Assemble the graph
    # Step 1: read in data from the .xls file
    xs, ys = load_fire_theft()

    xs = (xs - np.mean(xs))/np.std(xs)
    ys = (ys - np.mean(ys))/np.std(ys)

    # Step 2: create placeholders for input X (number of fire) and label Y (number of theft)
    X = tf.placeholder(tf.float32)
    Y = tf.placeholder(tf.float32)

    # Step 3, 4: create weights and predictions 
    with tf.name_scope("Model"):
        Y_pred = tf.Variable(0.0, name='Y_hat')
        for pow_i in range(0, POL_DEG):
            W = tf.Variable(tf.random_normal([1]), name='weight_%d' % pow_i)
            Y_pred = tf.add(tf.multiply(tf.pow(X, pow_i), W), Y_pred)

    # Step 5: use the square error as the loss function
    with tf.name_scope("Loss"):
        # Normal loss
        loss = tf.reduce_sum(tf.pow(Y_pred - Y, 2)) / (len(ys) - 1)
        # Regularization term
        loss = tf.add(loss, tf.multiply(LAMBDA, tf.global_norm([W])))

    # Step 6: using gradient descent with learning rate of 0.01 to minimize loss
    with tf.name_scope("SDG"):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    # Create operation to initialize all variables
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        # Step 7: initialize the necessary variables, in this case, w and b
        sess.run(init)

        # Step 8: train the model
        for epoch_i in range(n_epochs):
            for (x, y) in zip(xs, ys):
                sess.run(optimizer, feed_dict={X: x, Y: y})

            if epoch_i % 100 == 0:
                training_loss = sess.run(loss, feed_dict={X: xs, Y: ys})
                print("Epoch {0}: {1}".format(epoch_i, training_loss))

        # calculate function
        Y_hat = Y_pred.eval(feed_dict={X: xs}, session=sess)

    # have to sort for line plot
    # plot the results
    new_x, new_y = zip(*sorted(zip(xs, Y_hat)))
    plt.plot(xs, ys, 'bo',   label='Real data')
    plt.plot(new_x, new_y, 'r',    label='Predicted data')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()

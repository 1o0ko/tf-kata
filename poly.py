"""Simple tutorial for using TensorFlow to compute polynomial regression.
Parag K. Mital, Jan. 2016"""
# %% Imports
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import xlrd

DATA_FILE = 'data/fire_theft.xls'
LAMBDA = 0.01
POL_DEG = 5
learning_rate = 0.01
n_epochs = 1000

def main():
    # load data
    book = xlrd.open_workbook(DATA_FILE, encoding_override='utf-8')
    sheet = book.sheet_by_index(0)
    data = np.asarray([sheet.row_values(i) for i in range(1, sheet.nrows)])
    n_observations = sheet.nrows - 1
    xs = data[:,0].reshape(n_observations, 1)
    ys = data[:,1].reshape(n_observations, 1)

    xs = (xs - np.mean(xs))/np.std(xs)
    ys = (ys - np.mean(ys))/np.std(ys)

    # %% tf.placeholders for the input and output of the network. Placeholders are
    # variables which we need to fill in when we are ready to compute the graph.
    X = tf.placeholder(tf.float32)
    Y = tf.placeholder(tf.float32)

    # %% Instead of a single factor and a bias, we'll create a polynomial function
    # of different polynomial degrees.  We will then learn the influence that each
    # degree of the input (X^0, X^1, X^2, ...) has on the final output (Y).
    Y_pred = tf.Variable(0.0, name='Y_hat')
    for pow_i in range(0, POL_DEG):
        W = tf.Variable(tf.random_normal([1]), name='weight_%d' % pow_i)
        Y_pred = tf.add(tf.multiply(tf.pow(X, pow_i), W), Y_pred)

    # %% Loss function will measure the distance between our observations
    # and predictions and average over them.
    cost = tf.reduce_mean(tf.pow(Y_pred - Y, 2))
    cost = tf.add(cost, tf.multiply(LAMBDA, tf.global_norm([W])))

    # %% Use gradient descent to optimize W,b
    # Performs a single step in the negative gradient
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

    with tf.Session() as sess:
        # Here we tell tensorflow that we want to initialize all
        # the variables in the graph so we can use them
        sess.run(tf.global_variables_initializer())

        # Fit all training data
        prev_training_cost = 0.0
        for epoch_i in range(n_epochs):
            for (x, y) in zip(xs, ys):
                sess.run(optimizer, feed_dict={X: x, Y: y})

            training_cost = sess.run(cost, feed_dict={X: xs, Y: ys})

            if epoch_i % 100 == 0:
                print("Epoch {0}: {1}".format(i, training_cost))

            # Allow the training to quit if we've reached a minimum
            if np.abs(prev_training_cost - training_cost) < 0.00001:
                break
            prev_training_cost = training_cost

        # calculate function
        Y_hat = Y_pred.eval(feed_dict={X: xs}, session=sess)

    new_x, new_y = zip(*sorted(zip(xs, Y_hat)))
    plt.plot(xs, ys, 'bo',   label='Real data')
    plt.plot(new_x, new_y, 'r',    label='Predicted data')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()

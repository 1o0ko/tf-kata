"""
Simple linear regression example in TensorFlow using normal equation
This program tries to predict the number of thefts from
the number of fire in the city of Chicago
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from data_helpers import load_fire_theft


def main():
    # Load data, and add bias
    x_data, y_data = load_fire_theft()
    x_data_with_bias = np.hstack([np.ones((len(x_data), 1)), x_data])

    # Set up constant for the normal equation
    X = tf.constant(x_data_with_bias, dtype=tf.float32, name="X")
    y = tf.constant(y_data, dtype=tf.float32, name="y")
    XT = tf.transpose(X)

    theta = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(XT, X)), XT), y)

    with tf.Session() as sess:
        theta_value = sess.run(theta)

    print("Found following parameters: {0}".format(theta_value))

    # plot the results
    X, Y = x_data, y_data

    # Calculate the model predictions
    Y_hat = np.dot(x_data_with_bias, theta_value)
    plt.plot(X, Y, 'bo', label='Real data')
    plt.plot(X, Y_hat, 'r', label='Predicted data')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()

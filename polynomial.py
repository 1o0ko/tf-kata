"""
Simple linear regression example in TensorFlow
This program tries to predict the number of thefts from
the number of fire in the city of Chicago
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import xlrd

DATA_FILE = 'data/fire_theft.xls'
LAMBDA = 0.01
POL_DEG = 5
N_EPOCHS = 5000

def poly_n(x, n = POL_DEG):
    x_new = np.zeros(shape=(x.shape[0], POL_DEG))
    for i in range(n):
        x_new[:, i:(n+1)] = np.power(x, i)
        x_new[:, i] = x_new[:, i] / np.max(x_new[:, i])
    return x_new


def main():
    # Phase 1: Assemble the graph
    # Step 1: read in data from the .xls file
    book = xlrd.open_workbook(DATA_FILE, encoding_override='utf-8')
    sheet = book.sheet_by_index(0)
    data = np.asarray([sheet.row_values(i) for i in range(1, sheet.nrows)])
    n_samples = sheet.nrows - 1
    x_data = data[:,0].reshape(n_samples, 1) / np.max(data[:,0])
    y_data = data[:,1].reshape(n_samples, 1) / np.max(data[:,1])


   # Step 2: create placeholders for input X (number of fire) and label Y (number of theft)
    X = tf.placeholder(tf.float32, shape=[None, POL_DEG], name = "X")
    Y = tf.placeholder(tf.float32, shape=[None, 1], name = "Y")

    # Step 3: create weight and bias, initialized to 0
    w = tf.Variable(tf.random_normal([POL_DEG, 1]), name='w')

    # Step 4: predict Y (number of theft) from the number of fire
    with tf.name_scope("Model"):
        Y_predicted = tf.matmul(X, w, name='Y_predicted')

    # Step 5: use the square error as the loss function
    with tf.name_scope("Loss"):
        # error loss
        loss = tf.reduce_mean(tf.square(Y - Y_predicted, name = 'loss'))
        # add regularizer term
        loss = loss + 0.5 * LAMBDA * tf.reduce_sum(w**2)

    # Step 6: using gradient descent with learning rate of 0.01 to minimize loss
    with tf.name_scope("SDG"):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)


    # Create operation to initialize all variables
    init = tf.global_variables_initializer()

    # Create a summary to monitor loss
    training_summary = tf.summary.scalar("loss", loss)

    # Phase 2: Train our model
    with tf.Session() as sess:
        writer = tf.summary.FileWriter('./graphs', sess.graph)
        # Step 7: initialize the necessary variables, in this case, w and b
        sess.run(init)

        # Step 8: train the model
        cost_history = []
        for i in range(N_EPOCHS): # run 10 epochs
            # Session runs optimizer to minimize loss and fetch the value of loss
            _, loss_, summary, = sess.run([optimizer, loss, training_summary],
                                          feed_dict={X: poly_n(x_data), Y: y_data})
            if i % 100 == 0:
                writer.add_summary(summary, i)
                print("Epoch {0}: {1}".format(i, loss_))

        # Step 9: output the values of w
        w_value = sess.run(w)

    # plot the results
    Y_hat = np.matmul(poly_n(x_data), w_value)
    plt.plot(x_data, y_data, 'bo',   label='Real data')
    plt.plot(x_data, Y_hat, 'ro',    label='Predicted data')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()

"""
Simple linear regression example in TensorFlow
This program tries to predict the number of thefts from
the number of fire in the city of Chicago
"""
import matplotlib.pyplot as plt
import tensorflow as tf

from data_helpers import load_fire_theft


def main():
    # Phase 1: Assemble the graph
    # Step 1: read in data from the .xls file
    x_data, y_data = load_fire_theft()

    # Step 2: create placeholders for input X (number of fire) and label Y
    # (number of theft)
    X = tf.placeholder(tf.float32, shape=[None, 1], name="X")
    Y = tf.placeholder(tf.float32, shape=[None, 1], name="Y")

    # Step 3: create weight and bias, initialized to 0
    w = tf.Variable(tf.zeros([1, 1]), name='w')
    b = tf.Variable(tf.zeros([1, 1]), name='b')

    # Step 4: predict Y (number of theft) from the number of fire
    with tf.name_scope("Model"):
        Y_predicted = tf.add(tf.matmul(X, w), b, name='Y_predicted')

    # Step 5: use the square error as the loss function
    with tf.name_scope("Loss"):
        loss = tf.reduce_mean(tf.square(Y - Y_predicted, name='loss'))

    # Step 6: using gradient descent with learning rate of 0.01 to minimize
    # loss
    with tf.name_scope("SDG"):
        optimizer = tf.train.GradientDescentOptimizer(
            learning_rate=0.001).minimize(loss)

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
        for i in range(20):  # run 10 epochs
            # Session runs optimizer to minimize loss and fetch the value of
            # loss
            _, loss_, summary, = sess.run([optimizer, loss, training_summary],
                                          feed_dict={X: x_data, Y: y_data})
            writer.add_summary(summary, i)

            print("Epoch {0}: {1}".format(i, loss_))

        # Step 9: output the values of w and b
        w_value, b_value = sess.run([w, b])

    # plot the results
    X, Y = x_data, y_data
    plt.plot(X, Y, 'bo', label='Real data')
    plt.plot(X, X * w_value + b_value, 'r', label='Predicted data')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()

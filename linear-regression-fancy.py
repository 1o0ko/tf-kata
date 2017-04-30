"""
Simple linear regression example in TensorFlow
This program tries to predict the number of thefts from
the number of fire in the city of Chicago

In this example we follow the OOP approach
"""
from utils import define_scope

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import xlrd

DATA_FILE = 'data/fire_theft.xls'


class LinearModel:

    def __init__(self):
        self.X = tf.placeholder(tf.float32, shape=[None, 1], name="X")
        self.Y = tf.placeholder(tf.float32, shape=[None, 1], name="Y")

        self.w = tf.Variable(tf.zeros([1, 1]), name='w')
        self.b = tf.Variable(tf.zeros([1, 1]), name='b')

        self.prediction
        self.optimize
        self.error

    @define_scope
    def prediction(self):
        return tf.add(tf.matmul(self.X, self.w), self.b, name='Y_predicted')

    @define_scope
    def error(self):
        return tf.reduce_mean(tf.square(self.Y - self.prediction, name='loss'))

    @define_scope
    def optimize(self):
        return tf.train.GradientDescentOptimizer(
            learning_rate=0.001
        ).minimize(self.error)


class Dataset:

    def __init__(self, path=DATA_FILE):
        book = xlrd.open_workbook(DATA_FILE, encoding_override='utf-8')
        sheet = book.sheet_by_index(0)
        data = np.asarray([sheet.row_values(i) for i in range(1, sheet.nrows)])
        n_samples = sheet.nrows - 1

        self.x_data = data[:, 0].reshape(n_samples, 1)
        self.y_data = data[:, 1].reshape(n_samples, 1)

    @property
    def x(self):
        return self.x_data

    @property
    def y(self):
        return self.y_data


class Trainer:

    def run(self, model, data):
        # Create operation to initialize all variables
        init = tf.global_variables_initializer()

        # Create a summary to monitor loss
        training_summary = tf.summary.scalar("loss", model.error)

        # Phase 2: Train our model
        with tf.Session() as sess:
            writer = tf.summary.FileWriter('./graphs', sess.graph)
            sess.run(init)

            for i in range(20):
                _, loss_, summary, = sess.run(
                    [model.optimize, model.error, training_summary],
                    feed_dict={model.X: data.x, model.Y: data.y})

                writer.add_summary(summary, i)

                print("Epoch {0}: {1}".format(i, loss_))

            # Step 9: output the values of w and b
            w_value, b_value = sess.run([model.w, model.b])

            return w_value, b_value


def main():
    # Create model
    model = LinearModel()

    # Fetch data
    data = Dataset()

    # Run and create trainer
    w_value, b_value = Trainer().run(model, data)

    # plot the results
    X, Y = data.x, data.y
    plt.plot(X, Y,  'bo',   label='Real data')
    plt.plot(X, X * w_value + b_value,  'r',    label='Predicted data')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()

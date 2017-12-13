"""
Usage: train.py PATH

Arguments:
    PATH    path to save model
"""
import os

import numpy as np
import tensorflow as tf

from docopt import docopt

def sample_data():
    return np.array([10.0]).reshape(1,1)


class  Model(object):
    def __init__(self, graph_def):
        self.session = tf.Session()

        # load model and session
        checkpoint_dir = os.path.join(os.path.dirname(graph_def), 'model')
        saver = tf.train.import_meta_graph(graph_def)
        saver.restore(self.session, checkpoint_dir)

        # Load placeholder for data
        self.X = self.session.graph.get_tensor_by_name("X:0")
        self.Y = self.session.graph.get_tensor_by_name('Model/Y_predicted:0')

    def predict(self, x):
        predictions = self.session.run(self.Y, feed_dict={self.X: x})
        return predictions


def predict(graph_def, x_data):
    checkpoint_dir = os.path.join(os.path.dirname(graph_def), 'model')
    saver = tf.train.import_meta_graph(graph_def)
    graph = tf.get_default_graph()

    # Load placeholder for data
    X = graph.get_tensor_by_name("X:0")

    # Load operation to run
    Y = graph.get_tensor_by_name('Model/Y_predicted:0')
    with tf.Session() as sess:
        saver.restore(sess, checkpoint_dir)
        y_predicted = sess.run(Y, feed_dict={X: x_data})

    return y_predicted


if __name__ == '__main__':
    args = docopt(__doc__)

    print("Using function to load model and predict")
    print(predict(args['PATH'], sample_data()))

    print("Using class to manage session and conduct inference")
    model = Model(args['PATH'])
    print(model.predict(sample_data()))
    print(model.predict(sample_data()))

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


def predict(path, x_data):
    saver = tf.train.import_meta_graph(os.path.join(path, 'model.meta'))
    graph = tf.get_default_graph()

    print([op.name for op in graph.get_operations() if op.name.startswith('theta')])
    # Load placeholder for data
    X = graph.get_tensor_by_name("theta_input:0")

    # Load operation to run
    Y = graph.get_tensor_by_name('theta/BiasAdd:0')
    W = graph.get_tensor_by_name('theta/kernel:0')
    B = graph.get_tensor_by_name('theta/bias:0')

    # If we used dropout or any other operation that have different behaviour 
    # in test time, we would have to fetch appropriate tensor  and set it value to 0.
    with tf.Session() as sess:
        saver.restore(sess, os.path.join(path, 'model'))
        y_predicted, w, b = sess.run([Y, W, B], feed_dict={X: x_data})

    return y_predicted, w, b


if __name__ == '__main__':
    args = docopt(__doc__)

    print("Using function to load model and predict")
    print("Y_hat: %0.2f, w_hat: %0.2f, b_hat: %0.2f" %
          predict(args['PATH'], sample_data()))

"""
Usage: train.py PATH

Arguments:
    PATH    path to save model
"""
import os

import numpy as np
import tensorflow as tf

from docopt import docopt

def main(args):
    graph_def = args['PATH']
    checkpoint_dir = os.path.join(os.path.dirname(graph_def), 'model')

    saver = tf.train.import_meta_graph(graph_def)
    graph = tf.get_default_graph()

    # Load placeholder for data
    X = graph.get_tensor_by_name("X:0")
    x_data = np.array([10.0]).reshape(1,1)

    # Load operation to run
    Y = graph.get_tensor_by_name('Model/Y_predicted:0')
    print(Y)
    with tf.Session() as sess:
        saver.restore(sess, checkpoint_dir)
        y_predicted = sess.run(Y, feed_dict={X: x_data})
        print(y_predicted)

if __name__ == '__main__':
    args = docopt(__doc__)
    main(args)

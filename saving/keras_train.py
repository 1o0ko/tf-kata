"""
Usage: train_keras.py PATH

Arguments:
    PATH    path to save model
"""
import os

import tensorflow as tf

from docopt import docopt
from keras import backend as K
from keras.layers.core import Dense
from keras.models import Sequential

from utils import gen_data


def main(args):
    theta = (3, 1)
    x_data, y_data = gen_data(100, *theta)

    model = Sequential()
    model.add(
        Dense(
            1,
            input_shape=(
                1,
            ),
            init='uniform',
            name='theta',
            activation='linear'))
    model.compile(loss='mse', optimizer='rmsprop')

    model.fit(x_data, y_data, epochs=500, verbose=0)
    theta_hat = model.get_layer("theta").get_weights()
    w_hat, b_hat = theta_hat

    print("w_hat: %0.2f, b_hat: %0.2f" % (w_hat, b_hat))
    print("w: %0.2f, b: %0.2f" % theta)

    saver = tf.train.Saver()
    saver.save(K.get_session(), os.path.join(args['PATH'], 'model'))


if __name__ == '__main__':
    args = docopt(__doc__)
    main(args)

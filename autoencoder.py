'''
A simple, two-layer autoencoder on MNIST without weight-tying
Usage: autoencoder.py [options]

Options:
    --num_input=<int>                       Size of the input layer
                                            [default: 784]
    --num_hidden_1=<int>                    Size of the fist hidden layer
                                            [default: 256]
    --num_hidden_2=<int>                    Size of the second hidden layer
                                            [default: 128]

    -a, --learning-rate=<float>             Learning rate
                                            [default: 0.01]
    -n, --num-steps=<int>                   Number of SGD steps
                                            [default: 30000]
    -b, --batch-size=<int>                  Batch size
                                            [default: 256]


    -d, --display-step=<int>                How many steps need to pass to
                                            display evaluation
                                            [default: 1000]
    --num_rows=<int>                        Number of rows for visualization
                                            [default: 4]
'''
import matplotlib.pyplot as plt

from itertools import product

import numpy as np
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
from typeopt import Arguments

tf.logging.set_verbosity(tf.logging.INFO)


def encoder(x, args):
    ''' build encoder '''
    with tf.variable_scope("encoder"):
        # Without initializer glorot_uniform_initializer will be used
        # 1st layer
        W_1 = tf.get_variable("W_1", shape=[args.num_input, args.num_hidden_1])
        b_1 = tf.get_variable("b_1", shape=[args.num_hidden_1])
        layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, W_1), b_1))

        # 2nd layer
        W_2 = tf.get_variable("W_2", shape=[args.num_hidden_1, args.num_hidden_2])
        b_2 = tf.get_variable("b_2", shape=[args.num_hidden_2])
        layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, W_2), b_2))

    return layer_2


def decoder(x, args):
    ''' build decoder without weight tying '''
    with tf.variable_scope("decoder"):
        # 1st layer
        W_1 = tf.get_variable("W_1", shape=[args.num_hidden_2, args.num_hidden_1])
        b_1 = tf.get_variable("b_1", shape=[args.num_hidden_1])
        layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, W_1), b_1))

        # 2nd layer
        W_2 = tf.get_variable("W_2", shape=[args.num_hidden_1, args.num_input])
        b_2 = tf.get_variable("b_2", shape=[args.num_input])
        layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, W_2), b_2))

    return layer_2


def visualize_reconstructions(x, x_hat, n=4):
    '''
    Plot orginal and reconstructed images
    '''
    # reshape outoput:
    x = x.reshape([n, n, 28, 28])
    x_hat = x_hat.reshape([n, n, 28, 28])

    canvas_orig = np.empty((28 * n, 28 * n))
    canvas_recon = np.empty((28 * n, 28 * n))

    for i, j in product(range(n), range(n)):
        canvas_orig[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = \
            x[i, j, :, :]
        canvas_recon[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = \
            x_hat[i, j, :, :]

    plt.figure(figsize=(n, n))
    plt.imshow(canvas_orig, origin="upper", cmap="gray")
    plt.title("Original Images")
    plt.show()

    plt.figure(figsize=(n, n))
    plt.imshow(canvas_recon, origin="upper", cmap="gray")
    plt.title("Reconstructions Images")
    plt.show()


def main(args):
    ''' run experiment '''
    tf.logging.info('loading data...')
    mnist = input_data.read_data_sets("./datasets/mnist", one_hot=True)

    tf.logging.info('Building model...')
    with tf.name_scope("Input"):
        X = tf.placeholder("float", [None, args.num_input])

    with tf.name_scope("Model"):
        codes = encoder(X, args)
        X_hat = decoder(codes, args)

    with tf.name_scope("Loss"):
        loss = tf.reduce_mean(tf.pow(X - X_hat, 2))

    with tf.name_scope("SGD"):
        optimizer = tf.train.AdamOptimizer(args.learning_rate).minimize(loss)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        for i in range(1, args.num_steps + 1):
            # Get batch
            batch_x, _ = mnist.train.next_batch(args.batch_size)

            # Run optimization step and get loss
            _, loss_val = sess.run([optimizer, loss], feed_dict={X: batch_x})

            if i % args.display_step == 0 or i == 1:
                tf.logging.info('Step %i: Minibatch Loss: %f' % (i, loss_val))

        # Visualize images
        sample_images, _ = mnist.test.next_batch(args.num_rows**2)
        reconstructed = sess.run(X_hat, feed_dict={X: sample_images})

        visualize_reconstructions(sample_images, reconstructed, args.num_rows)


if __name__ == '__main__':
    main(Arguments(__doc__, version='Autoencoder example 0.1'))

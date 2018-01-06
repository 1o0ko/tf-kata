import time
import numpy as np
import tensorflow as tf

from tensorflow.contrib.layers import xavier_initializer


def highway_1(x, i=1, carry_bias=-2.0):
    D = x.shape[-1].value
    with tf.variable_scope("highway_1_%d" % i):
        # Srivastava et al. (2015) recommend initializing bT to a negative
        # value, in order to militate the initial behavior towards carry.
        W_T = tf.get_variable(
            "W_T", shape=[D, D],
            initializer=xavier_initializer())

        b_T = tf.get_variable(
            "b_T", shape=[D],
            initializer=tf.random_uniform_initializer(
                minval=carry_bias - 0.2,
                maxval=carry_bias + 0.2))

        W_H = tf.get_variable(
            "W_H", shape=[D, D], 
            initializer=xavier_initializer())
        b_H = tf.get_variable(
            "b_H", shape=[D],
            initializer=tf.constant_initializer(0.1))

        H = tf.nn.relu(
            tf.einsum('ijk,kl->ijl', x, W_H) + b_H,
            name="activation")
        T = tf.sigmoid(
            tf.einsum('ijk,kl->ijl', x,W_T) + b_T,
            name="transform_gate")
        C = tf.subtract(1.0, T, name="carry_gate")

        y = tf.add(tf.multiply(H, T), tf.multiply(x, C), "y")

    return y


def highway_2(x, i=1, carry_bias=-2.0):
    D = x.shape[-1].value
    with tf.variable_scope("highway_2_%d" % i):
        H = tf.layers.dense(x, D, activation=tf.nn.relu,
                            kernel_initializer=xavier_initializer(),
                            bias_initializer=tf.constant_initializer(0.1),
                            name="activation")

        # Srivastava et al. (2015) recommend initializing bT to a negative
        # value, in order to militate the initial behavior towards carry.
        T = tf.layers.dense(x, D, activation=tf.nn.sigmoid,
                            kernel_initializer=xavier_initializer(),
                            bias_initializer=tf.random_uniform_initializer(
                                minval=carry_bias - 0.2,
                                maxval=carry_bias + 0.2),
                            name="transform_gate")
        C = tf.subtract(1.0, T, name="carry_gate")

    return tf.add(tf.multiply(H, T), tf.multiply(x, C))


def highway_3(x, i=1, carry_bias=-2.0):
    D, idx = x.shape[-1].value, len(x.shape) - 1
    with tf.variable_scope("highway_3_%d" % i):
        # Srivastava et al. (2015) recommend initializing bT to a negative
        # value, in order to militate the initial behavior towards carry.
        W_T = tf.get_variable(
            "W_T", shape=[D, D], 
            initializer=xavier_initializer())
        b_T = tf.get_variable(
            "b_T", shape=[D],
            initializer=tf.random_uniform_initializer(
                minval=carry_bias - 0.2,
                maxval=carry_bias + 0.2))

        W_H = tf.get_variable(
            "W_H", shape=[D, D], 
            initializer=xavier_initializer())
        b_H = tf.get_variable(
            "b_H", shape=[D],
            initializer=tf.constant_initializer(0.1))

        H = tf.nn.relu(tf.tensordot(x, W_H, [[idx], [0]]) + b_H, 
                       name="activation")
        T = tf.sigmoid(tf.tensordot(x, W_T, [[idx], [0]]) + b_T, 
                       name="transform_gate")
        C = tf.subtract(1.0, T, name="carry_gate")

        y = tf.add(tf.multiply(H, T), tf.multiply(x, C), "y")

    return y


class HighwayBenchmark(tf.test.Benchmark):

    def benchmarkLayer(self, layer_fn, name, B=5, T=10, D=2, iters=100):

        x = tf.placeholder(tf.float32, shape=(None, T, D))
        x_val = np.random.random((B, T, D))
        h = layer_fn(x)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            start_time = time.time()
            for _ in range(iters):
                sess.run(h, feed_dict={
                    x: x_val
                })

            total_wall_time = time.time() - start_time

            self.report_benchmark(
                name="wall_time [%s]" % name,
                wall_time=total_wall_time / iters,
                iters=iters,
                extras={
                    'total_wall_time': total_wall_time
                })


if __name__ == "__main__":
    print("Setting up benchamrks")
    B, T, D, iters = 32, 30, 30, 11

    benchmark = HighwayBenchmark()
    benchmark.benchmarkLayer(highway_1, "einsum", B, T, D, iters)
    benchmark.benchmarkLayer(highway_2, "layers.dense", B, T, D, iters)
    benchmark.benchmarkLayer(highway_3, "tf.tensordot", B, T, D, iters)

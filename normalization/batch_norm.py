'''
Batch Normalization implementation in tensorflow:
    "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift" (https://arxiv.org/abs/1502.03167)
'''
import tensorflow as tf


class batch_norm(object):
    """Code modification of http://stackoverflow.com/a/33950177"""

    def __init__(self, epsilon=1e-5, momentum=0.1, name="batch_norm"):
        with tf.variable_scope(name):
            self.epsilon = epsilon
            self.momentum = momentum

            self.ema = tf.train.ExponentialMovingAverage(decay=self.momentum)
            self.name = name

    def __call__(self, x, train=True):
        shape = x.get_shape().as_list()

        with tf.variable_scope(self.name):
            self.gamma = tf.get_variable(
                "gamma", [shape[-1]], initializer=tf.random_normal_initializer(1., 0.02))
            self.beta = tf.get_variable(
                "beta", [shape[-1]], initializer=tf.constant_initializer(0.))

            mean, variance = tf.nn.moments(x, [0, 1, 2])

            return tf.nn.batch_norm_with_global_normalization(
                x, mean, variance, self.beta, self.gamma, self.epsilon,
                scale_after_normalization=True)


if __name__ == '__main__':
    print("yo!")

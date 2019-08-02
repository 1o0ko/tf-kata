"""Tests for nucleus sampling ops. """
import unittest
import numpy as np
import tensorflow as tf

from nlp.sampling.nucleus import nucleus_logits


class TestNucleus(unittest.TestCase):
    def test_logits(self):
        logits = tf.constant([
            [3.0, 0.1, 2.5, -.1, -.5],
            [0.8, 0.1, 0.5, 0.5, 2.0],
            [0.1, 0.1, 0.2, 0.6, 0.3]])
        
        with tf.compat.v1.Session() as sess:
            n_logits = sess.run(nucleus_logits(logits, 0.8))

        np.allclose(n_logits, np.array([
            [ 3. , -np.inf,  2.5, -np.inf, -np.inf],
            [ 0.8, -np.inf,  0.5,  0.5,  2. ],
            [ 0.1,  0.1,  0.2,  0.6,  0.3]]))

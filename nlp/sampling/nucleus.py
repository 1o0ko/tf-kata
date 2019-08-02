"""Implementation of nucleus sampling algorithm from 'The Curious Case of Neural Text Degeneration' (arXiv:1904.09751) paper.

"""
import numpy as np
import tensorflow as tf


def nucleus_logits(logits: tf.Tensor, probablity_threshold: float) -> tf.Tensor:
    """Select logits that accounts for selected probablity mass.
    
    We select the highest probability tokens whose cumulative probability mass
    exceeds our pre-chosen threshold p.
    """
    n_batches = tf.shape(logits)[0]
    
    logits_sorted = tf.sort(logits, axis=-1, direction='DESCENDING')
    
    cummulative_probabilities = tf.cumsum(tf.nn.softmax(logits_sorted, axis=-1), axis = -1)
    
    cutoff_indices  = tf.reduce_sum(tf.cast(
        cummulative_probabilities < probablity_threshold, dtype=tf.int32), axis=-1)

    threshold_indices = tf.stack([tf.range(n_batches), cutoff_indices], axis=-1)
    threshold_logits = tf.expand_dims(tf.gather_nd(logits_sorted, threshold_indices), axis=-1)
    
    return tf.compat.v2.where(logits >= threshold_logits, logits, -np.inf)

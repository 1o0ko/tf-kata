'''
Character level word-embedding from 'Character-Aware Neural Language Models'
https://arxiv.org/pdf/1508.06615.pdf
'''
from highway import highway
import tensorflow as tf


class CharCNNEmbedding(object):
    '''
    Character level word-embedding
    '''

    def __init__(self, input_, max_words, max_chars,
                 vocab_size, char_emb_size,
                 filter_sizes, num_filters,
                 number_of_highway_layers, aux_tokens=2):

        # Inputs
        self.input = input_
        self.max_words = max_words
        self.max_chars = max_chars
        self.aux_tokens = aux_tokens

        # General
        self.number_of_highway_layers = number_of_highway_layers

        # CNN
        self.char_emb_size = char_emb_size
        self.vocab_size = vocab_size
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.output_dim = num_filters * len(filter_sizes)

        cnn_outputs = []
        with tf.variable_scope("char_embedding", reuse=tf.AUTO_REUSE) as scope:
            # create char embedding matrix
            nrows = self.vocab_size - self.aux_tokens
            ncols = self.char_emb_size
            E = tf.get_variable(
                "embedding_matrix",
                shape=[nrows, ncols],
                initializer=tf.random_uniform_initializer(-1.0, 1.0))

            # add zeros for word with  0 and 1 indices (PAD and UNK)
            E = tf.concat([
                tf.zeros([self.aux_tokens, self.char_emb_size]), E], axis=0
            )

            # get list of character indices
            char_indices = tf.split(self.input, self.max_words, axis=1)
            for idx, char_index in enumerate(char_indices):
                '''
                If we're processing the second sentence in the batch, we must
                reuse variables (such as CNN filters created in
                `convolve_embedded_chars`)
                '''
                if idx != 0:
                    scope.reuse_variables()
                char_index = tf.reshape(char_index, [-1, self.max_chars])

                # create word matrix for convolution
                chars_emb = tf.nn.embedding_lookup(E, char_index)

                # apply filters on the word matrix
                chars_cnn = self.convolve_embedded_chars(chars_emb, idx != 0)

                # apply highway layer
                for i in range(self.number_of_highway_layers):
                    chars_cnn = highway(chars_cnn, i=i)

                # NOTE:
                # It's worth pointing out that, if we have highway layers and use bias in
                # non-linearity after convolution operation, i.e. tanh(<C, K> + b) then
                # the we'll have non-zero elements in zero padded sentences.
                # We need some sort of masking to account for that.

                # adding additional dimension to later perform concatenation on
                cnn_outputs.append(tf.reshape(
                    chars_cnn, [-1, 1, self.output_dim]))

        # we want to return a tensor od shape [batch_size, number of words,
        # word embedding size]
        self.output = tf.concat(cnn_outputs, axis=1)

    def convolve_embedded_chars(self, embedded_chars, reuse_variables):
        '''
        Obtains character level word representation
        '''
        # conv2d operation expects a 4-dimensional tensor [batch_size x
        # seq_length x embed_dim x 1]
        embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)

        pooled_outputs = []
        for i, filter_size in enumerate(self.filter_sizes):
            with tf.variable_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [
                    filter_size,
                    self.char_emb_size, 1, self.num_filters]

                W = tf.get_variable(
                    "W", shape=filter_shape,
                    initializer=tf.truncated_normal_initializer(stddev=0.1))

                b = tf.get_variable(
                    "b", shape=[self.num_filters],
                    initializer=tf.constant_initializer(0.1))

                conv = tf.nn.conv2d(
                    embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")

                # Apply non-linearity
                f = tf.nn.tanh(tf.nn.bias_add(conv, b), name="tanh")

                # Want to reduce the convolved image" size to 1
                f_height = f.shape[1].value

                # Max-pooling over time
                pooled = tf.nn.max_pool(
                    f,
                    ksize=[1, f_height, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        h_pool = tf.concat(pooled_outputs, axis=3)

        # [batch_size, 1, 1, N] -> [batch_size, N]
        h_pool_flat = tf.squeeze(h_pool, [1, 2])

        return h_pool_flat

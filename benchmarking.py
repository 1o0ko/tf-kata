'''
Trivial example of tf profiling.

After runing the script fo to chrome://tracing and load the timeline.json

More could be find here:
    * tf docs:      https://www.tensorflow.org/get_started/graph_viz#runtime_statistics
    * blog post:    https://towardsdatascience.com/howto-profile-tensorflow-1a49fb18073d
    * github code:  https://github.com/ikhlestov/tensorflow_profiling
'''
import os
import tensorflow as tf

from tensorflow.python.client import timeline

OUT_DIR = './graphs'


def benchmark_op(fetches, name):
    with tf.Session() as sess:
        # Create writer to dump profiling data to
        writer = tf.summary.FileWriter(os.path.join(OUT_DIR, name), sess.graph)

        # Record execution stats
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()

        # Run the graph with full trace option
        sess.run(fetches, options=run_options, run_metadata=run_metadata)

        # Add metadata to the writer
        writer.add_run_metadata(run_metadata, 'steps', global_step=None)

        # Create the Timeline object, and write it to a json
        tl = timeline.Timeline(run_metadata.step_stats)
        ctf = tl.generate_chrome_trace_format()
        with open('timeline_%s.json' % name, 'w') as f:
            f.write(ctf)


if __name__ == '__main__':
    with tf.name_scope("training_ops"):
        x = tf.random_normal([1000, 4000], name="x")
        y = tf.random_normal([4000, 1000], name="y")
        with tf.name_scope('fine_grained'):
            z = tf.add(y, y, name='adding')
            a = tf.nn.relu(z, name='ReLU')
            res = tf.matmul(x, a, name="result")

    benchmark_op(res, name='matmul')

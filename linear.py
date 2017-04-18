'''
Simple linear reggression in Tensorflow
'''
import tensorflow as tf


def main():
    ''' Build graph and run it'''

    a = tf.constant([2, 3], name='a')
    b = tf.constant([2, 5], name='b')
    x = tf.add(a, b)

    with tf.Session() as sess:
        writer = tf.summary.FileWriter('./graphs', sess.graph)
        print(sess.run(x))

    # close the writer when you’re done using it
    writer.close()

if __name__ == '__main__':
    main()

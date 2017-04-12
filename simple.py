'''
Simple script showcasting Tensorboard
'''
import tensorflow as tf


def main():
    ''' Build graph and run it'''

    a = tf.constant(2)
    b = tf.constant(3)
    x = tf.add(a, b)

    with tf.Session() as sess:
        writer = tf.summary.FileWriter('./graphs', sess.graph)
        print(sess.run(x))

    # close the writer when you’re done using it
    writer.close()

if __name__ == '__main__':
    main()

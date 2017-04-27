'''
Brief example of how to load custom data structures into tensorflow
'''
import os
import random
import tensorflow as tf

from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes


DATASET_PATH='/data/mnist'
TEST_LABELS_FILE='test-labels.csv'
TRAIN_LABELS_FILE='train-labels.csv'

TEST_SET_SIZE = 5
IMAGE_HEIGHT  = 28
IMAGE_WIDTH   = 28
NUM_CHANNELS  = 3
BATCH_SIZE    = 5

def encode_label(label):
    return int(label)

def read_label_file(source_file):
    with open(source_file, "r") as f:
        filepaths, labels = [], []
        for line in f:
            filepath, label = line.split(",")
            filepaths.append(filepath)
            labels.append(encode_label(label))

    return filepaths, labels


if __name__ == '__main__':
    train_filepaths, train_labels = read_label_file(os.path.join(DATASET_PATH, TRAIN_LABELS_FILE))
    test_filepaths, test_labels = read_label_file(os.path.join(DATASET_PATH, TEST_LABELS_FILE))

    # full path
    train_filepaths = [os.path.join(DATASET_PATH, file_path) 
        for file_path in train_filepaths]

    test_filepaths =  [os.path.join(DATASET_PATH, file_path) 
        for file_path in test_filepaths]

    # join and take sampe
    all_filepaths = (train_filepaths + test_filepaths)[:20]
    all_labels = (train_labels + test_labels)[:20]

    print(all_labels)
    print(all_filepaths)

    # convert string into tensors
    all_images = ops.convert_to_tensor(all_filepaths, dtype=dtypes.string)
    all_labels = ops.convert_to_tensor(all_labels, dtype=dtypes.int32)

    # create a partition vector
    partitions = [0] * len(all_filepaths)
    partitions[:TEST_SET_SIZE] = [1] * TEST_SET_SIZE
    random.shuffle(partitions)

    # partition our data into a test and train set according to our partition vector
    train_images, test_images = tf.dynamic_partition(all_images, partitions, 2)
    train_labels, test_labels = tf.dynamic_partition(all_labels, partitions, 2)

    # create input queues
    train_input_queue = tf.train.slice_input_producer(
					[train_images, train_labels],
					shuffle=False)
    test_input_queue = tf.train.slice_input_producer(
					[test_images, test_labels],
					shuffle=False)

    # process path and string tensor into an image and a label
    file_content = tf.read_file(train_input_queue[0])
    train_image = tf.image.decode_jpeg(file_content, channels=NUM_CHANNELS)
    train_label = train_input_queue[1]

    file_content = tf.read_file(test_input_queue[0])
    test_image = tf.image.decode_jpeg(file_content, channels=NUM_CHANNELS)
    test_label = test_input_queue[1]

    # define tensor shape
    train_image.set_shape([IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS])
    test_image.set_shape([IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS])


    # collect batches of images before processing
    train_image_batch, train_label_batch = tf.train.batch(
					[train_image, train_label],
					batch_size=BATCH_SIZE
					#,num_threads=1
					)
    test_image_batch, test_label_batch = tf.train.batch(
					[test_image, test_label],
					batch_size=BATCH_SIZE
					#,num_threads=1
					)

    print "input pipeline ready"

    with tf.Session() as sess:

      # initialize the variables
      sess.run(tf.initialize_all_variables())

      # initialize the queue threads to start to shovel data
      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(coord=coord)

      print "from the train set:"
      for i in range(20):
	print sess.run(train_label_batch)

      print "from the test set:"
      for i in range(10):
	print sess.run(test_label_batch)

      # stop our queue threads and properly close the session
      coord.request_stop()
      coord.join(threads)
      sess.close()

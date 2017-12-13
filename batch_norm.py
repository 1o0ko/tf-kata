'''
Batch Normalization implementation in tensorflow:
    "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift" (https://arxiv.org/abs/1502.03167)
'''
import numpy as np
import tensorflow as tf
import tqdm 

from tensorflow.examples.tutorials.mnist import input_data


if __name__ == '__main__':
    mnist = input_data.read_data_sets("./datasets/mnist/", one_hot=True)
    print("yo!")

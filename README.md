# 🥋 tf-kata 🥋
Repository that is losely inspired by [CS 20SI: Tensorflow for Deep Learning Research](http://web.stanford.edu/class/cs20si/)

# Models
## Linear regression
* [linear regression using normal equation](./linear-regression-normal-equation.py)
* [linear regression with SGD](./linear-regression.py)
* [linear regression with polynomial kernel](./polynomial-regression.py): linear regression with `polynomial kernel` aka `polynomial regression`.
* [linear regression with OOP](./linear-regression-fancy.py): simple example how to organize code with classes. 

## Logistic regression
* [logistic regression](./logistic-regression.py) trained on `obligatory` MNIST.

## Autoencoders
* [Simple 2-layer autoencoder without weight-tying](./autoencoder.py) trained on `obligatory` MNIST.
* [Variational Auto-Encoder using tf.contrib.distributionss](./vae.py) trained on `obligatory` MNIST.

## CNN
* [Classification with CNN on MNIST using Dataset, Estimator and Experiment API](./higher-level-apis.py)

## Layers
* [highway layer](./highway.py) ['Highway Networks'](http://arxiv.org/abs/1505.00387) paper

## NLP Embeddings
* [charcter-level CNN word embeddings](./nlp/char-cnn.py) from [Character Aware Language Models](https://arxiv.org/pdf/1508.06615.pdf)

# Infrastructure and Debugging
## Tensorboard
* [minimal example](./minimal-tensorboard.py)
* [logistic regression with train and validation split](./logistic-regression.py): validation and training error plotted on the same graph. 
* [ops benchmarking with tensorboard and Timeline object](./bencharking.py)

## Pipelines
* [custom datastructures](./pipelines/pipeline-simple.py): MNIST as a custom dataset

## Model managament
* [training and saving](./saving/train.py): Train simple linear reggresion and save it.
* [loading saved model](./saving/load.py): Load placeholder and ouput tensor and use it for prediction.
* [training and saving in Keras](./saving/keras_train.py): Train simple linear reggresion in Keras and save it.
* [loading saved model in Keras](./saving/keras_load.py): Load model and use it in production. 
* [exporting a clean graph](./saving/graph_freeze.py): Select sub-graph relevant to operations, freeze weights and export definition
* [loading and serving a frozen graph](./saving/graph_serve.py): Load the graph and serve it using Falcon API endpoint


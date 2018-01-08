'''
Dataset, Estimator, Experiment APIS
https://gist.github.com/peterroelants/9956ec93a07ca4e9ba5bc415b014bcca
'''
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data as mnist_data
from tensorflow.contrib import slim
from tensorflow.contrib.learn import ModeKeys
from tensorflow.contrib.learn import learn_runner


tf.logging.set_verbosity(tf.logging.INFO)

# Set define default flags
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string(
    flag_name='model_dir', default_value='./datasets/mnist/mnist_training',
    docstring='Output directory for model and training stats.')
tf.app.flags.DEFINE_string(
    flag_name='data_dir', default_value='./datasets/mnist/mnist_data',
    docstring='Directory to download the data to.')
tf.app.flags.DEFINE_integer(
    flag_name='batch_size', default_value=128,
    docstring='Size of batch.')


# Definie and run the experiment 
def run_experiment(argsv=None):
    ''' Run the training experiment '''
    # Define the model parameters
    params = tf.contrib.training.HParams(
        learning_rate=0.002,
        n_classes=10,
        training_steps=5000,
        min_eval_frequency=100,
        batch_size=FLAGS.batch_size
    )

    # Set the run_config and the directory to save the model and stats
    run_config = tf.contrib.learn.RunConfig(
        model_dir=FLAGS.model_dir
    )

    learn_runner.run(
        experiment_fn=experiment_fn,    # First-class function
        run_config=run_config,          # RunConfig
        schedule="train_and_evaluate",  # What to run
        hparams=params
    )


def experiment_fn(run_config, params):
    '''
    Create an experiment to train and evaluate the model

    Args:
        run_config (RunConfig): Configuration for Estimator run.
        params (HParam): Hyperparameters
    Returns:
        (Experiment) Experiment for training the mnist model.
    '''
    tf.logging.info(run_config)
    tf.logging.info(params)

    # You can change a subset of the run_config properties as
    run_config = run_config.replace(
        save_checkpoints_steps=params.min_eval_frequency)

    # Define the mnist classifier
    estimator = get_estimator(run_config, params)

    # Setup data loaders
    mnist = mnist_data.read_data_sets(FLAGS.data_dir, one_hot=False)

    train_input_fn, train_input_hook = get_train_inputs(
        batch_size=params.batch_size,
        mnist_data=mnist)

    eval_input_fn, eval_input_hook = get_test_inputs(
        batch_size=params.batch_size,
        mnist_data=mnist)

    # Define the experiment
    experiment = tf.contrib.learn.Experiment(
        estimator=estimator,                            # Estimator
        train_input_fn=train_input_fn,                  # First-class function
        eval_input_fn=eval_input_fn,                    # First-class function
        train_steps=params.train_steps,                 # Minibatch steps
        min_eval_frequency=params.min_eval_frequency,   # Eval frequency
        train_monitors=[train_input_hook],              # Hooks for training
        eval_hooks=[eval_input_hook],                   # Hooks for evaluation
        eval_steps=None                                 # Use evaluation feeder until its empty
    )

    return experiment

# Define model ############################################
def get_estimator(run_config, params):
    """
    Return the model as a Tensorflow Estimator object.

    Args:
         run_config (RunConfig): Configuration for Estimator run.
         params (HParams): hyperparameters.
    """
    return tf.estimator.Estimator(
        model_fn=model_fn,      # First-class function
        params=params,          # HParams
        config=run_config       # RunConfig
    )


def model_fn(features, labels, mode, params):
    pass


def get_train_op_fn(loss, params):
    pass


def get_eval_metric_ops(labels, predictions):
    pass


def architecture(inputs, is_training, scope='MnistConvNet'):
    pass


# Define data loaders #####################################
class IteratorInitializerHook(tf.train.SessionRunHook):
    pass


# Define the training inputs
def get_train_inputs(batch_size, mnist_data):
    pass


# Define the training inputs
def get_train_inputs(batch_size, mnist_data):
    pass


def get_test_inputs(batch_size, mnist_data):
    pass


if __name__ == '__main__':
    tf.app.run(
        main=run_experiment
    )

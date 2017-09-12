# coding=utf-8
import tensorflow as tf

from datasets import dataset_factory
from deployment import model_deploy
from nets import nets_factory
from preprocessing import preprocessing_factory

slim = tf.contrib.slim

tf.app.flags.DEFINE_string('master', '', 'The address of the TensorFlow master to use')
tf.app.flags.DEFINE_string('train_dir', r'D:\tf_log', 'Directory where checkpoints and event logs are written to.')

tf.app.flags.DEFINE_integer('num_clones', 1, 'Number of model clones to deploy')
tf.app.flags.DEFINE_boolean('clone_on_cpu', False, 'Use CPUs to deploy clones')
tf.app.flags.DEFINE_integer('worker_replicas', 1, 'Number of worker replicas')
tf.app.flags.DEFINE_integer('num_ps_tasks', 0, 'The number of parameter servers. If the value is 0, then the parameters'
                                               'are handle locally by the worker')

tf.app.flags.DEFINE_integer('num_readers', 4, 'The number of parallel readers that read data from the dataset.')
tf.app.flags.DEFINE_integer('num_preprocessing_threads', 4, 'The number of threads used to create the batches')

tf.app.flags.DEFINE_integer('log_every_n_steps', 10, 'The frequency with which logs are print.')
tf.app.flags.DEFINE_integer('save_summaries_secs', 600, 'The frequency with which summaries are saved, in seconds.')
tf.app.flags.DEFINE_integer('save_interval_secs', 600, 'The frequency with which the model is saved, in seconds.')

tf.app.flags.DEFINE_integer('task', 0, 'Task id of the replica running the training.')

######################
# Optimization Flags #
######################

tf.app.flags.DEFINE_float('weight_decay', 0.00004, 'The weight on the model weights')
tf.app.flags.DEFINE_string('optimizer', 'sgd', 'the name of the optimizer, one of "adadelta", '
                                               '"adagrad", "adam", "ftrl", "momentum", "sgd" or "rmsprop"')
tf.app.flags.DEFINE_float('adadelta_rho', 0.95, 'the decay rate for adadelta.')


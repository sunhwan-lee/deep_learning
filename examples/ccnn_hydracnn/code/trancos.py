# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Builds the CIFAR-10 network.

Summary of available functions:

 # Compute input images and labels for training. If you would like to run
 # evaluations, use inputs() instead.
 inputs, labels = distorted_inputs()

 # Compute inference on the model inputs to make a prediction.
 predictions = inference(inputs)

 # Compute the total loss of the prediction with respect to the labels.
 loss = loss(predictions, labels)

 # Create a graph to run one step of training with respect to the loss.
 train_op = train(loss, global_step)
"""
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import sys
import tarfile

from six.moves import urllib
import tensorflow as tf

import numpy as np

FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_string('data_dir', '../data/TRANCOS/images/',
                           """Path to TRANCOS data directory.""")
tf.app.flags.DEFINE_string('train_feat_list', '../output/features/train.txt',
                           """Path to the list of TRANCOS training features.""")
tf.app.flags.DEFINE_integer('image_size', 72*72*3,
                            """Number of pixels in input image""")
tf.app.flags.DEFINE_integer('den_size', 18*18,
                            """Number of pixels in density map""")


# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999      # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0       # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.001 # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.0001     # Initial learning rate.

# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'

def _activation_summary(x):
  """Helper to create summaries for activations.

  Creates a summary that provides a histogram of activations.
  Creates a summary that measures the sparsity of activations.

  Args:
    x: Tensor
  Returns:
    nothing
  """
  # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
  # session. This helps the clarity of presentation on tensorboard.
  tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
  tf.summary.histogram(tensor_name + '/activations', x)
  tf.summary.scalar(tensor_name + '/sparsity',
                                       tf.nn.zero_fraction(x))


def _variable_on_cpu(name, shape, initializer):
  """Helper to create a Variable stored on CPU memory.

  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable

  Returns:
    Variable Tensor
  """
  with tf.device('/cpu:0'):
    dtype = tf.float32
    var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
  return var


def _variable_with_weight_decay(name, shape, stddev, wd):
  """Helper to create an initialized Variable with weight decay.

  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.

  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian If None, constant 
            initializer
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.

  Returns:
    Variable Tensor
  """
  dtype = tf.float32
  var = _variable_on_cpu(
      name,
      shape,
      tf.truncated_normal_initializer(stddev=stddev, dtype=dtype) if stddev is not None 
        else tf.constant_initializer(0.0))
  if wd is not None:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
  return var

def read_and_decode(filename_queue):
  
  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(filename_queue)
  features = tf.parse_single_example(
      serialized_example,
      # Defaults are not specified since both keys are required.
      features={
          'image_raw': tf.FixedLenFeature([], tf.string),
          'label_raw': tf.FixedLenFeature([], tf.string),
      })

  # Convert from a scalar string tensor to a float32 tensor with shape [FLAGS.image_size].
  image = tf.decode_raw(features['image_raw'], tf.float32)
  image.set_shape([FLAGS.image_size])
  #image.set_shape([4*4*3])

  # Convert label from a scalar string tensor to float32 tensor with
  # shape [FLAGS.den_size].
  label = tf.decode_raw(features['label_raw'], tf.float32)
  label.set_shape([FLAGS.den_size])
  #label.set_shape([2*2])

  return image, label

def inputs(batch_size, shuffle=False):
  """Reads input data num_epochs times.
  Args:
    batch_size: Number of examples per returned batch.
  Returns:
    A tuple (images, labels), where:
    * images is a float tensor with shape [batch_size, IMAGE_PIXELS].
    * labels is a float tensor with shape [batch_size, LABEL_PIXELS].
    Note that an tf.train.QueueRunner is added to the graph, which
    must be run using e.g. tf.train.start_queue_runners().
  """
  if not FLAGS.train_feat_list:
    raise ValueError('Please supply a train_feat_list')

  filenames = np.loadtxt(FLAGS.train_feat_list, dtype='str')
  filenames = ["../"+f for f in filenames]  
  
  with tf.name_scope('input'):
    filename_queue = tf.train.string_input_producer(filenames, num_epochs=1)

    # Even when reading in multiple threads, share the filename queue.
    image, label = read_and_decode(filename_queue)
    
    # Shuffle the examples and collect them into batch_size batches.
    if shuffle:
      images, labels = tf.train.shuffle_batch(
          [image, label], batch_size=batch_size, num_threads=2,
          capacity=1000 + 3 * batch_size,
          # Ensures a minimum amount of shuffling of examples.
          min_after_dequeue=1000)
    else:
      images, labels = tf.train.batch(
          [image, label], batch_size=batch_size, num_threads=2,
          capacity=1000 + 3 * batch_size)

    return images, labels

def inference(images):
  """Build the TRANCOS model.

  Args:
    images: Images returned from inputs().

  Returns:
    Output of final conv layer with 18x18x1
  """
  # We instantiate all variables using tf.get_variable() instead of
  # tf.Variable() in order to share variables across multiple GPU training runs.
  # If we only ran this model on a single GPU, we could simplify this function
  # by replacing all instances of tf.get_variable() with tf.Variable().
  #
  # conv1
  with tf.variable_scope('conv1') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[7, 7, 3, 32],
                                         stddev=1e-2,
                                         wd=1e-3)
    conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [32], tf.constant_initializer(0.0))
    pre_activation = tf.nn.bias_add(conv, biases)
    conv1 = tf.nn.relu(pre_activation, name=scope.name)
    _activation_summary(conv1)

  # pool1
  pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool1')

  # conv2
  with tf.variable_scope('conv2') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[7, 7, 32, 32],
                                         stddev=3e-1,
                                         wd=1e-3)
    conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [32], tf.constant_initializer(0.0))
    pre_activation = tf.nn.bias_add(conv, biases)
    conv2 = tf.nn.relu(pre_activation, name=scope.name)
    _activation_summary(conv2)

  # pool2
  pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool2')

  # conv3
  with tf.variable_scope('conv3') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[5, 5, 32, 64],
                                         stddev=1e-3,
                                         wd=1e-3)
    conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
    pre_activation = tf.nn.bias_add(conv,biases)
    conv3 = tf.nn.relu(pre_activation, name=scope.name)
    _activation_summary(conv3)

  # conv4
  with tf.variable_scope('conv4') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[1, 1, 64, 1000],
                                         stddev=3e-1,
                                         wd=1e-3)
    conv = tf.nn.conv2d(conv3, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [1000], tf.constant_initializer(0.0))
    pre_activation = tf.nn.bias_add(conv,biases)
    conv4 = tf.nn.relu(pre_activation, name=scope.name)
    _activation_summary(conv4)

  # conv5
  with tf.variable_scope('conv5') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[1, 1, 1000, 400],
                                         stddev=1e-1,
                                         wd=1e-3)
    conv = tf.nn.conv2d(conv4, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [400], tf.constant_initializer(0.0))
    pre_activation = tf.nn.bias_add(conv,biases)
    conv5 = tf.nn.relu(pre_activation, name=scope.name)
    _activation_summary(conv5)

  # conv6
  with tf.variable_scope('conv6') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[1, 1, 400, 1],
                                         stddev=None,
                                         wd=1e-3)
    conv = tf.nn.conv2d(conv5, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [1], tf.constant_initializer(0.0))
    pre_activation = tf.nn.bias_add(conv,biases)
    conv6 = tf.nn.relu(pre_activation, name=scope.name)
    _activation_summary(conv6)

  return conv6

def loss(labels, pred):
  """Add L2Loss to all the trainable variables.

  Add summary for "Loss" and "Loss/avg".
  Args:
    logits: Logits from inference().
    labels: Labels from distorted_inputs or inputs(). 1-D tensor
            of shape [batch_size]

  Returns:
    Loss tensor of type float.
  """
  # Calculate the Euclidean loss across the batch.
  batch_size = tf.cast(labels.get_shape()[0], tf.float32)
  euclidean_loss = tf.nn.l2_loss(pred - labels)
  euclidean_loss_mean = tf.divide(euclidean_loss, batch_size, name='euclidean_loss')
  tf.add_to_collection('losses', euclidean_loss_mean)

  return tf.add_n(tf.get_collection('losses'), name='total_loss')

def _add_loss_summaries(total_loss):
  """Add summaries for losses in TRANCOS model.

  Generates moving average for all losses and associated summaries for
  visualizing the performance of the network.

  Args:
    total_loss: Total loss from loss().
  Returns:
    loss_averages_op: op for generating moving averages of losses.
  """
  # Compute the moving average of all individual losses and the total loss.
  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  losses = tf.get_collection('losses')
  loss_averages_op = loss_averages.apply(losses)

  # Attach a scalar summary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.
  #for l in losses + [total_loss]:
  for l in losses:
    # Name each loss as '(raw)' and name the moving average version of the loss
    # as the original loss name.
    tf.summary.scalar(l.op.name + ' (raw)', l)
    tf.summary.scalar(l.op.name, loss_averages.average(l))

  return loss_averages_op


def train(total_loss, global_step):
  """Train TRANCOS model.

  Create an optimizer and apply to all trainable variables. Add moving
  average for all trainable variables.

  Args:
    total_loss: Total loss
    global_step: Integer Variable counting the number of training steps
      processed.
  Returns:
    train_op: op for training.
  """
  # Variables that affect learning rate.
  decay_steps = 1

  # Decay the learning rate exponentially based on the number of steps.
  lr = tf.train.inverse_time_decay(INITIAL_LEARNING_RATE,
                                   global_step,
                                   decay_steps,
                                   LEARNING_RATE_DECAY_FACTOR)
  tf.summary.scalar('learning_rate', lr)

  # Generate moving averages of all losses and associated summaries.
  loss_averages_op = _add_loss_summaries(total_loss)

  # Compute gradients.
  with tf.control_dependencies([loss_averages_op]):
    opt = tf.train.GradientDescentOptimizer(lr)
    grads = opt.compute_gradients(total_loss)

  # Apply gradients.
  apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

  # Add histograms for trainable variables.
  for var in tf.trainable_variables():
    tf.summary.histogram(var.op.name, var)

  # Add histograms for gradients.
  for grad, var in grads:
    if grad is not None:
      tf.summary.histogram(var.op.name + '/gradients', grad)

  # Track the moving averages of all trainable variables.
  variable_averages = tf.train.ExponentialMovingAverage(
      MOVING_AVERAGE_DECAY, global_step)
  variables_averages_op = variable_averages.apply(tf.trainable_variables())

  with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
    train_op = tf.no_op(name='train')

  return train_op

def maybe_download_and_extract():
  """Download and extract the tarball from Alex's website."""
  dest_directory = FLAGS.data_dir
  if not os.path.exists(dest_directory):
    os.makedirs(dest_directory)
  filename = DATA_URL.split('/')[-1]
  filepath = os.path.join(dest_directory, filename)
  if not os.path.exists(filepath):
    def _progress(count, block_size, total_size):
      sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
          float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()
    filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
    print()
    statinfo = os.stat(filepath)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
  extracted_dir_path = os.path.join(dest_directory, 'cifar-10-batches-bin')
  if not os.path.exists(extracted_dir_path):
    tarfile.open(filepath, 'r:gz').extractall(dest_directory)
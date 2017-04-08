"""Builds the MNIST network.

Implements the inference/loss/training pattern for model building.

1. inference() - Builds the model as far as is required for running the network forward to make predictions.
2. loss() - Adds to the inference model the layers required to generate loss.
3. training() - Adds to the loss model the Ops required to generate and apply gradients.

This file is used by the various "fully_connected_*.py" files and not meant to
be run.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# The MNIST dataset has 10 classes, representing the digits 0 through 9.
NUM_CLASSES = 10

# The MNIST images are always 28x28 pixels.
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE

def inference(images, hidden1_units, hidden2_units):

  """
  Args:
    images: Images placeholder
    hidden1_units: Size of the first hidden layer
    hidden2_units: Size of the second hidden layer

  Returns:
    softmax_linear: Output tensor with computed logits
  """

  with tf.name_scope("hidden1"):
    weights = tf.Variable(tf.truncated_normal([IMAGE_PIXELS, hidden1_units], 
                                              stddev = 1.0 / math.sqrt(float(IMAGE_PIXELS))),
                          name = "weights")
    biases  = tf.Variable(tf.zeros([hidden1_units]), name = "biases")
    hidden1 = tf.nn.relu(tf.matmul(images, weights) + biases)

  with tf.name_scope("hidden2"):
    weights = tf.Variable(tf.truncated_normal([hidden1_units, hidden2_units], 
                                              stddev = 1.0 / math.sqrt(float(hidden1_units))),
                          name = "weights")
    biases  = tf.Variable(tf.zeros([hidden2_units]), name = "biases")
    hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)

  with tf.name_scope("softmax_linear"):
    weights = tf.Variable(tf.truncated_normal([hidden2_units, NUM_CLASSES], 
                                              stddev = 1.0 / math.sqrt(float(hidden2_units))),
                          name = "weights")
    biases  = tf.Variable(tf.zeros([NUM_CLASSES]), name = "biases")
    logits = tf.matmul(hidden2, weights) + biases

  return logits

def loss(logits, labels):
  """
  Args:
    logits: logits from the prediction, float - [batch size, NUM_CLASSES]
    labels: Actual labels,              int32 - [batch_size]

  Returns:
    loss: Loss function to be minimized
  """

  labels = tf.to_int64(labels)
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits, name="xentropy")

  return tf.reduce_mean(cross_entropy, name="xentropy_mean")

def training(loss, learning_rate):
  """
  Creates a summarizer to track the loss over time in TensorBoard.

  Creates an optimizer and applies the gradients to all trainable variables.

  The Op returned by this function is what must be passed to the `sess.run()` call to cause the model to train.

  Args:
    loss: Loss tensor of type float
    learning_rate: Learning rate of type float

  Returns:
    train_op: An Operation that updates the variables
  """

  tf.summary.scalar("loss", loss)
  global_step = tf.Variable(0, name="global_step", trainable=False)
  optimizer = tf.train.GradientDescentOptimizer(learning_rate)

  train_op = optimizer.minimize(loss, global_step = global_step)

  return train_op

def evaluation(logits, labels):
  """
  Evaluate the accuracy of the trained model

  Args:
    logits: Logits tensor, float - [batch size, NUM_CLASSES]
    labels: Labels tensor, int32 - [batch size]

  Returns:
    A scalar int32 tensor with the number of examples (out of batch_size)
    that were predicted correctly.
  """

  correct_prediction = tf.nn.in_top_k(logits, labels, 1)

  return tf.reduce_sum(tf.cast(correct_prediction, tf.float32))



import tensorflow as tf
import npy_to_tfrecord as nttf

import numpy as np
# create 10 sample labels and image
# labels: array of 10 x 2 x 2
# image: array of 10 x 4 x 4 x 3

labels = np.random.rand(3200,18,18).astype(np.float32)
#labels = np.random.randint(10, size=(10,2,2))
images = np.random.rand(3200,72,72,3).astype(np.float32)

print labels[0,:]
print images[0,:]
#nttf.npy_to_tfr(images, labels, "../output/features/trancos_train_feat_sample1")
nttf.npy_to_tfr(images, labels, "../output/features/trancos_train_feat_sample0")

labels = np.random.rand(3200,18,18).astype(np.float32)
images = np.random.rand(3200,72,72,3).astype(np.float32)

print labels[0,:]
print images[0,:]
#nttf.npy_to_tfr(images, labels, "../output/features/trancos_train_feat_sample2")
nttf.npy_to_tfr(images, labels, "../output/features/trancos_train_feat_sample1")

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

  # Convert from a scalar string tensor (whose single string has
  # length mnist.IMAGE_PIXELS) to a uint8 tensor with shape
  # [mnist.IMAGE_PIXELS].
  image = tf.decode_raw(features['image_raw'], tf.float32)
  #  #image = tf.cast(features['image_raw'], tf.float32)
  image.set_shape([72*72*3])

  # OPTIONAL: Could reshape into a 28x28 image and apply distortions
  # here.  Since we are not applying any distortions in this
  # example, and the next step expects the image to be flattened
  # into a vector, we don't bother.

  # Convert from [0, 255] -> [-0.5, 0.5] floats.
  #image = tf.cast(image, tf.float32) * (1. / 255) - 0.5

  # Convert label from a scalar uint8 tensor to an int32 scalar.
  label = tf.decode_raw(features['label_raw'], tf.float32)
  label.set_shape([18*18])

  return image, label

def inputs(batch_size, num_epochs):
  """Reads input data num_epochs times.
  Args:
    train: Selects between the training (True) and validation (False) data.
    batch_size: Number of examples per returned batch.
    num_epochs: Number of times to read the input data, or 0/None to
       train forever.
  Returns:
    A tuple (images, labels), where:
    * images is a float tensor with shape [batch_size, mnist.IMAGE_PIXELS]
      in the range [-0.5, 0.5].
    * labels is an int32 tensor with shape [batch_size] with the true label,
      a number in the range [0, mnist.NUM_CLASSES).
    Note that an tf.train.QueueRunner is added to the graph, which
    must be run using e.g. tf.train.start_queue_runners().
  """
  if not num_epochs: num_epochs = None
  
  filename = ["../output/features/trancos_train_feat_sample"+str(i)+".tfrecords" for i in range(2)]

  with tf.name_scope('input'):
    filename_queue = tf.train.string_input_producer(
        filename, num_epochs=num_epochs)

    # Even when reading in multiple threads, share the filename
    # queue.
    image, label = read_and_decode(filename_queue)
    
    # Shuffle the examples and collect them into batch_size batches.
    # (Internally uses a RandomShuffleQueue.)
    # We run this in two threads to avoid being a bottleneck.
    #images, labels = tf.train.shuffle_batch(
    #    [image, label], batch_size=batch_size, num_threads=2,
    #    capacity=1000 + 3 * batch_size,
        # Ensures a minimum amount of shuffling of examples.
    #    min_after_dequeue=1000)
    images, labels = tf.train.batch(
        [image, label], batch_size=batch_size, num_threads=2,
        capacity=1000 + 3 * batch_size)

    return images, labels

with tf.Graph().as_default():
  # Input images and labels.
  images, labels = inputs(batch_size=5,
                          num_epochs=1)

  # The op for initializing the variables.
  init_op = tf.group(tf.global_variables_initializer(),
                     tf.local_variables_initializer())

  # Create a session for running operations in the Graph.
  sess = tf.Session()

  # Initialize the variables (the trained variables and the
  # epoch counter).
  sess.run(init_op)

  # Start input enqueue threads.
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(sess=sess, coord=coord)

  im = sess.run([images, labels])
  print(im[0],im[1])

  # Wait for threads to finish.
  #coord.join(threads)
  #sess.close()

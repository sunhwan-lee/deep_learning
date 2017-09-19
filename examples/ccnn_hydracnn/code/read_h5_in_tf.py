import tensorflow as tf
import tftables


#filename_queue = tf.train.string_input_producer(tf.train.match_filenames_once("../output/features/trancos_train_feat_*.h5"))

def distorted_inputs():

  filename_queue = ["../output/features/trancos_train_feat_" + str(i) + ".h5" for i in range(3)]
  for filename in filename_queue:

    print "file name:", filename
    reader = tftables.open_file(filename=filename, batch_size=100)

    # Use get_batch to access the table.
    # Both datasets must be accessed in ordered mode.
    label_batch = reader.get_batch(
        path = '/label',
        ordered = True)

    # Now use get_batch again to access an array.
    # Both datasets must be accessed in ordered mode.
    data_batch = reader.get_batch('/data_s0', ordered = True)
    key_batch = reader.get_batch('/key', ordered = True)

    # The loader takes a list of tensors to be stored in the queue.
    # When accessing in ordered mode, threads should be set to 1.
    loader = reader.get_fifoloader(
        queue_size = 10,
        inputs = [key_batch, label_batch, data_batch],
        threads = 1)

    return loader

    # Batches are taken out of the queue using a dequeue operation.
    # Tensors are returned in the order they were given when creating the loader.
    #label_cpu, data_cpu = loader.dequeue()

    #return label_cpu, data_cpu

with tf.device('/cpu:0'):
  loader = distorted_inputs()
  keys, labels, images = loader.dequeue()

# The dequeued data can then be used in your network.
#result = my_network(label_cpu, data_cpu)

#initialize the variable
#init_op = tf.initialize_all_variables()

N=100
with tf.Session() as sess:
#  sess.run(init_op)
  with loader.begin(sess):
    for n in range(N):
      print(n)
      print(sess.run(tf.shape(labels)))
      print(sess.run(keys))

#reader.close()

import h5py
filename = "../output/features/trancos_train_feat_0.h5"
f = h5py.File(filename, 'r')

# List all groups
print(list(f["key"]))
print()

filename = "../output/features/trancos_train_feat_1.h5"
f = h5py.File(filename, 'r')
# List all groups
print(list(f["key"]))
print()

filename = "../output/features/trancos_train_feat_2.h5"
f = h5py.File(filename, 'r')
# List all groups
print(list(f["key"]))


import tensorflow as tf
import tftables

reader = tftables.open_file(filename="../output/features/trancos_train_feat_0.h5", batch_size=10)

label_batch = reader.get_batch(
    path = '/label',
    cyclic = True,
    ordered = True
)

data_batch = reader.get_batch(
    path = '/data_s0',
    cyclic = True,
    ordered = True
)

# Now we create a FIFO Loader
loader = reader.get_fifoloader(
    queue_size = 10,              			# The maximum number of elements that the
                                  			# internal Tensorflow queue should hold.
    inputs = [label_batch, data_batch], # A list of tensors that will be stored
                                  			# in the queue.
		threads = 1						              # The number of threads used to stuff the
				                                # queue. If ordered access to a dataset
				                                # was requested, then only 1 thread
				                                # should be used.
)

# Batches can now be dequeued from the loader for use in your network.
array_batch_cpu = loader.dequeue()
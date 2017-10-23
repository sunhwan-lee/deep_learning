from utils.data_manager import DataManager
from utils.neural_network import NeuralNetwork
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
import pickle
import datetime
import time
import os

tf.flags.DEFINE_string('data_dir', 'data',
                       'Data directory containing nostalgic and ordinary narratives, \'wordsList.npy\', \'wordVectors.npy\'.'
                       ' Intermediate files will automatically be stored here')
tf.flags.DEFINE_string('stopwords_file', 'data/stopwords.txt',
                       'Path to stopwords file. If stopwords_file is None, no stopwords will be used')
tf.flags.DEFINE_integer('n_samples', None,
                        'Number of samples to use from the dataset. Set n_samples=None to use the whole dataset')
tf.flags.DEFINE_string('checkpoints_root', 'checkpoints',
                       'Checkpoints directory. Parameters will be saved there')
tf.flags.DEFINE_string('summaries_dir', 'logs',
                       'Directory where TensorFlow summaries will be stored')
tf.flags.DEFINE_integer('batch_size', 16,
                        'Batch size')
tf.flags.DEFINE_integer('train_steps', 300,
                        'Number of training steps')
tf.flags.DEFINE_integer('hidden_size', 75,
                        'Hidden size of LSTM layer')
tf.flags.DEFINE_integer('embedding_size', 50,
                        'Size of embeddings layer')
tf.flags.DEFINE_integer('random_state', 0,
                        'Random state used for data splitting. Default is 0')
tf.flags.DEFINE_float('learning_rate', 0.01,
                      'RMSProp learning rate')
tf.flags.DEFINE_float('test_size', 0.2,
                      '0<test_size<1. Proportion of the dataset to be included in the test split.')
tf.flags.DEFINE_float('dropout_keep_prob', 0.75,
                      '0<dropout_keep_prob<=1. Dropout keep-probability')
tf.flags.DEFINE_integer('sequence_len', None,
                        'Maximum sequence length. Let m be the maximum sequence length in the'
                        ' dataset. Then, it\'s required that sequence_len >= m. If sequence_len'
                        ' is None, then it\'ll be automatically assigned to m')
tf.flags.DEFINE_integer('validate_every', 100,
                        'Step frequency in order to evaluate the model using a validation set')
FLAGS = tf.flags.FLAGS

# Prepare summaries
summaries_dir = '{0}/{1}'.format(FLAGS.summaries_dir,
                                 datetime.datetime.now().strftime('%d_%b_%Y-%H_%M_%S'))
train_writer = tf.summary.FileWriter(summaries_dir + '/train')
validation_writer = tf.summary.FileWriter(summaries_dir + '/validation')

# Prepare model directory
#model_name = str(int(time.time()))
#model_dir = '{0}/{1}'.format(FLAGS.checkpoints_root, model_name)
#if not os.path.exists(model_dir):
#    os.makedirs(model_dir)

# Save configuration
FLAGS._parse_flags()
config = FLAGS.__dict__['__flags']
with open('{}/config.pkl'.format(summaries_dir + '/train'), 'wb') as f:
    pickle.dump(config, f)
#with open('{}/config.pkl'.format(model_dir), 'wb') as f:
#    pickle.dump(config, f)

# Prepare data and build TensorFlow graph
dm = DataManager(data_dir=FLAGS.data_dir,
                 stopwords_file=FLAGS.stopwords_file,
                 sequence_len=FLAGS.sequence_len,
                 test_size=FLAGS.test_size,
                 val_samples=FLAGS.batch_size,
                 n_samples=FLAGS.n_samples,
                 random_state=FLAGS.random_state)

# create tsv file for word embedding visualiation in tensorboard
vocab_sorted = sorted([(k,v[0],v[1]) for k,v in dm._vocab.items()], key=lambda x:x[1])
with open(os.path.join(summaries_dir, "metadata_" + str(FLAGS.n_samples) + ".tsv"), 'wb') as f:
  f.write('Word\tFrequency\n' + '\n'.join([(w[0] if len(w[0]) else 'NONE') + '\t' + str(w[2]) for w in vocab_sorted]))

nn = NeuralNetwork(hidden_size=[FLAGS.hidden_size],
                   vocab_size=dm.vocab_size,
                   embedding_size=FLAGS.embedding_size,
                   max_length=dm.sequence_len,
                   learning_rate=FLAGS.learning_rate)

project_config = projector.ProjectorConfig()
embedding = project_config.embeddings.add()
embedding.tensor_name = nn.embeddings.name

# Link this tensor to its metadata file (e.g. labels).
embedding.metadata_path = os.path.join(summaries_dir, "metadata_" + str(FLAGS.n_samples) + ".tsv")

# The next line writes a projector_config.pbtxt in the LOG_DIR. TensorBoard will
# read this file during startup.
projector.visualize_embeddings(train_writer, project_config)

# Train model
sess = tf.Session()
sess.run(nn.initialize_all_variables())
saver = tf.train.Saver()
x_val, y_val, val_seq_len = dm.get_val_data()
train_writer.add_graph(nn.input.graph)

for i in range(FLAGS.train_steps):
    # Perform training step
    x_train, y_train, train_seq_len = dm.next_batch(FLAGS.batch_size)
    train_loss, _, summary = sess.run([nn.loss, nn.train_step, nn.merged],
                                      feed_dict={nn.input: x_train,
                                                 nn.target: y_train,
                                                 nn.seq_len: train_seq_len,
                                                 nn.dropout_keep_prob: FLAGS.dropout_keep_prob})
    train_writer.add_summary(summary, i)  # Write train summary for step i (TensorBoard visualization)
    print('{0}/{1} train loss: {2:.4f}'.format(i + 1, FLAGS.train_steps, train_loss))

    # Check validation performance
    if (i + 1) % FLAGS.validate_every == 0:
        val_loss, accuracy, summary = sess.run([nn.loss, nn.accuracy, nn.merged],
                                               feed_dict={nn.input: x_val,
                                                          nn.target: y_val,
                                                          nn.seq_len: val_seq_len,
                                                          nn.dropout_keep_prob: 1})
        validation_writer.add_summary(summary, i)  # Write validation summary for step i (TensorBoard visualization)
        print('   validation loss: {0:.4f} (accuracy {1:.4f})'.format(val_loss, accuracy))

# Save model
checkpoint_file = '{}/model.ckpt'.format(summaries_dir + '/train')
save_path = saver.save(sess, checkpoint_file)
print('Model saved in: {0}'.format(summaries_dir + '/train'))



#checkpoint_file = '{}/model.ckpt'.format(model_dir)
#save_path = saver.save(sess, checkpoint_file)
#print('Model saved in: {0}'.format(model_dir))

import tensorflow as tf
from utils.data_manager import DataManager
import pickle
from sklearn.metrics import roc_auc_score, roc_curve, auc

tf.flags.DEFINE_string('checkpoints_dir', 'checkpoints',
                       'Checkpoints directory (example: checkpoints/1479670630). Must contain (at least):\n'
                       '- config.pkl: Contains parameters used to train the model \n'
                       '- model.ckpt: Contains the weights of the model \n'
                       '- model.ckpt.meta: Contains the TensorFlow graph definition \n')
FLAGS = tf.flags.FLAGS

if FLAGS.checkpoints_dir is None:
    raise ValueError('Please, a valid checkpoints directory is required (--checkpoints_dir <file name>)')

# Load configuration
with open('{}/config.pkl'.format(FLAGS.checkpoints_dir), 'rb') as f:
    config = pickle.load(f)

# Load data
dm = DataManager(data_dir=config['data_dir'],
                 stopwords_file=config['stopwords_file'],
                 sequence_len=config['sequence_len'],
                 n_samples=config['n_samples'],
                 test_size=config['test_size'],
                 val_samples=config['batch_size'],
                 random_state=config['random_state'],
                 ensure_preprocessed=True)

# Import graph and evaluate the model using test data
original_text, x_test, y_test, test_seq_len = dm.get_test_data(original_text=True)
graph = tf.Graph()
with graph.as_default():
    sess = tf.Session()

    # Import graph and restore its weights
    print('Restoring graph ...')
    saver = tf.train.import_meta_graph("{}/model.ckpt.meta".format(FLAGS.checkpoints_dir))
    saver.restore(sess, ("{}/model.ckpt".format(FLAGS.checkpoints_dir)))

    # Recover input/output tensors
    input = graph.get_operation_by_name('input').outputs[0]
    target = graph.get_operation_by_name('target').outputs[0]
    seq_len = graph.get_operation_by_name('lengths').outputs[0]
    dropout_keep_prob = graph.get_operation_by_name('dropout_keep_prob').outputs[0]
    predict = graph.get_operation_by_name('final_layer/softmax/predictions').outputs[0]
    accuracy = graph.get_operation_by_name('accuracy/accuracy').outputs[0]

    # Perform prediction
    pred, acc = sess.run([predict, accuracy],
                         feed_dict={input: x_test,
                                    target: y_test,
                                    seq_len: test_seq_len,
                                    dropout_keep_prob: 1})

# Print results
for i in range(len(original_text)):
    print('Sample: {0}'.format(original_text[i]))
    print('Predicted sentiment: [{0:.4f}, {1:.4f}]'.format(pred[i, 0], pred[i, 1]))
    print('Real sentiment: {0}\n'.format(y_test[i]))
print('\nAccuracy: {0:.4f}'.format(acc))
print('\nAUC score: {0:.4f}'.format(roc_auc_score(y_test, pred)))

import matplotlib.pyplot as plt
# Compute ROC curve and ROC area for each class
fpr, tpr, threshold = roc_curve(y_test[:,1], pred[:,1])
roc_auc = auc(fpr, tpr)

plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.savefig('roc_curve_nostalgic.pdf')
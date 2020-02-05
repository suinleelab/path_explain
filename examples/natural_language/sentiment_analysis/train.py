import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from preprocess import sentiment_dataset
from model import cnn_model
from path_explain.utils import set_up_environment

from absl import app
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_integer('batch_size', 128, 'Model batch size')
flags.DEFINE_integer('num_epochs', 2, 'Number of epochs to train for')
flags.DEFINE_integer('sequence_length', 52, 'Maximum length of any sequence')
flags.DEFINE_integer('hidden_units', 50, 'Number of hidden units')
flags.DEFINE_integer('num_filters',  32, 'Number of convolutional filters of each size')
flags.DEFINE_integer('embedding_dim', 32, 'Size of the embedding')
flags.DEFINE_float('dropout_rate', 0.5, 'Fraction of inputs to set to 0 during training')
flags.DEFINE_float('learning_rate', 0.001, 'Learning rate to use while training')
flags.DEFINE_string('visible_devices', '0', 'CUDA_VISIBLE_DEVICES flag')
flags.DEFINE_boolean('eval_only', False, 'Set to true to evaluate a stored model')

def train(argv=None):
    set_up_environment(visible_devices=FLAGS.visible_devices)

    print("Loading data...")
    train_set, vald_set = sentiment_dataset(batch_size=FLAGS.batch_size,
                                            max_sequence_length=FLAGS.sequence_length)
    encoder = tfds.features.text.TokenTextEncoder.load_from_file('encoder')


    if FLAGS.eval_only:
        model = tf.keras.models.load_model('model.h5')
        print('Evaluating on the training set...')
        model.evaluate(train_set, verbose=1)
        print('Evaluating on the validation set...')
        model.evaluate(vald_set, verbose=1)
        return

    print('Building model...')
    model = cnn_model(encoder.vocab_size,
                      FLAGS.embedding_dim,
                      FLAGS.sequence_length,
                      FLAGS.dropout_rate,
                      FLAGS.num_filters,
                      FLAGS.hidden_units)

    model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                  optimizer=tf.keras.optimizers.Adam(learning_rate=FLAGS.learning_rate),
                  metrics=[tf.keras.metrics.BinaryAccuracy(),
                           tf.keras.metrics.AUC()])

    model.fit(train_set,
              epochs=FLAGS.num_epochs,
              validation_data=vald_set,
              verbose=1)
    tf.keras.models.save_model(model, 'model.h5')

if __name__ == '__main__':
    app.run(train)
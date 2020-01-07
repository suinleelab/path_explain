"""
Train convolutional network for sentiment analysis on IMDB corpus. Based on
"Convolutional Neural Networks for Sentence Classification" by Yoon Kim
http://arxiv.org/pdf/1408.5882v2.pdf and the GitHub repository:
https://github.com/alexander-rakhlin/CNN-for-Sentence-Classification-in-Keras
"""

import tensorflow as tf
import numpy as np
from model import cnn_model, lstm_model

from absl import app
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_integer('sequence_length', 400, 'Maximum length of any sequence')
flags.DEFINE_integer('max_words', 20000, 'Maximum number of index words in the vocab')
flags.DEFINE_integer('batch_size', 128, 'Model batch size')
flags.DEFINE_integer('num_epochs', 2, 'Number of epochs to train for')
flags.DEFINE_integer('hidden_units', 50, 'Number of hidden units')
flags.DEFINE_integer('num_filters',  32, 'Number of convolutional filters of each size')
flags.DEFINE_integer('embedding_dim', 32, 'Size of the embedding')
flags.DEFINE_float('dropout_rate', 0.5, 'Fraction of inputs to set to 0 during training')
flags.DEFINE_float('learning_rate', 0.001, 'Learning rate to use while training')
flags.DEFINE_string('model_type', 'cnn', 'One of `lstm`, `cnn`.')


def load_data(max_words, sequence_length):
    index_from = 3
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=max_words,
                                                                            index_from=index_from)

    x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train,
                                                            maxlen=sequence_length,
                                                            padding="post",
                                                            truncating="post")
    x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test,
                                                           maxlen=sequence_length,
                                                           padding="post",
                                                           truncating="post")

    vocabulary = tf.keras.datasets.imdb.get_word_index()
    vocabulary_inv = dict((v + index_from, k) for k, v in vocabulary.items())
    vocabulary_inv[0] = "<PAD/>"
    vocabulary_inv[1] = '<START/>'
    vocabulary_inv[2] = '<UNKNOWN/>'
    return x_train, y_train, x_test, y_test, vocabulary_inv

def train(argv=None):
    print("Loading data...")
    x_train, y_train, x_test, y_test, vocabulary_inv = load_data(FLAGS.max_words,
                                                                 FLAGS.sequence_length)

    print('Building model...')
    if FLAGS.model_type == 'cnn':
        model = cnn_model(len(vocabulary_inv),
                          FLAGS.embedding_dim,
                          FLAGS.sequence_length,
                          FLAGS.dropout_rate,
                          FLAGS.num_filters,
                          FLAGS.hidden_units)
    elif FLAGS.model_type == 'lstm':
        model = lstm_model(vocab_length=len(vocabulary_inv),
                           embedding_dim=FLAGS.embedding_dim,
                           sequence_length=FLAGS.sequence_length,
                           dropout_rate=FLAGS.dropout_rate,
                           lstm_units=FLAGS.num_filters,
                           hidden_units=FLAGS.hidden_units)
    else:
        raise ValueError('Unrecognized value `{}` for argument model_type'.format(FLAGS.model_type))

    if FLAGS.sequence_length != x_test.shape[1]:
        print("Adjusting sequence length for actual size")
        FLAGS.sequence_length = x_test.shape[1]

    print("x_train shape:", x_train.shape)
    print("x_test shape:", x_test.shape)
    print("Vocabulary Size: {:d}".format(len(vocabulary_inv)))

    model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                  optimizer=tf.keras.optimizers.Adam(learning_rate=FLAGS.learning_rate),
                  metrics=[tf.keras.metrics.BinaryAccuracy(),
                           tf.keras.metrics.AUC()])

    model.fit(x_train,
              y_train,
              batch_size=FLAGS.batch_size,
              epochs=FLAGS.num_epochs,
              validation_data=(x_test, y_test),
              verbose=1)
    tf.keras.models.save_model(model, '{}.h5'.format(FLAGS.model_type))

if __name__ == '__main__':
    app.run(train)
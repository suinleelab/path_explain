import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm

from path_explain.path_explainer_tf import PathExplainerTF
from path_explain import utils
from train import load_data
from model import cnn_model, lstm_model

from absl import app
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_integer('num_samples', 300, 'Number of samples to draw for attributions')
flags.DEFINE_integer('num_sentences', 100, 'Number of sentences to interpret')
flags.DEFINE_string('visible_devices', '0', 'GPU to use')

def interpret(argv=None):
    print('Setting up environment...')
    utils.set_up_environment(visible_devices=FLAGS.visible_devices)

    print('Loading data...')
    x_train, y_train, x_test, y_test, vocabulary_inv = load_data(FLAGS.max_words, FLAGS.sequence_length)

    lengths = np.sum(x_test != 0, axis=1)
    min_indices = np.argsort(lengths)

    print('Loading model...')
    if FLAGS.model_type == 'cnn':
        interpret_model = cnn_model(len(vocabulary_inv),
                                    FLAGS.embedding_dim,
                                    FLAGS.sequence_length,
                                    FLAGS.dropout_rate,
                                    FLAGS.num_filters,
                                    FLAGS.hidden_units,
                                    for_interpretation=True)
    elif FLAGS.model_type == 'lstm':
        interpret_model = lstm_model(vocab_length=len(vocabulary_inv),
                                     embedding_dim=FLAGS.embedding_dim,
                                     sequence_length=FLAGS.sequence_length,
                                     dropout_rate=FLAGS.dropout_rate,
                                     lstm_units=FLAGS.num_filters,
                                     hidden_units=FLAGS.hidden_units,
                                     for_interpretation=True)
    else:
        raise ValueError('Unrecognized value `{}` for argument model_type'.format(FLAGS.model_type))


    model = tf.keras.models.load_model('{}.h5'.format(FLAGS.model_type))
    embedding_model = tf.keras.models.Model(model.input, model.layers[1].output)

    interpret_model.load_weights('{}.h5'.format(FLAGS.model_type), by_name=True)

    explainer = PathExplainerTF(interpret_model)

    batch_input = x_test[min_indices[:FLAGS.num_sentences]]
    batch_embedding = embedding_model(batch_input)
    batch_pred = model(batch_input)

    baseline_input = np.zeros(x_test[0:1].shape)
    baseline_embedding = embedding_model(baseline_input)

    print('Getting attributions...')
    # Get word-level attributions
    embedding_attributions = explainer.attributions(batch_embedding,
                                                    baseline_embedding,
                                                    batch_size=FLAGS.batch_size,
                                                    num_samples=FLAGS.num_samples,
                                                    use_expectation=False,
                                                    output_indices=0,
                                                    verbose=True)
    np.save('embedding_attributions_{}.npy'.format(FLAGS.model_type),
            embedding_attributions)

    print('Getting interactions...')
    # Get pairwise word interactions
    max_indices = np.sum(batch_input[-1] != 0)
    interaction_matrix = np.zeros((FLAGS.num_sentences,
                                   max_indices,
                                   FLAGS.embedding_dim,
                                   FLAGS.sequence_length,
                                   FLAGS.embedding_dim))

    indices = np.indices((max_indices,
                          FLAGS.embedding_dim))
    indices = indices.reshape(2, -1)
    indices = indices.swapaxes(0, 1)
    for interaction_index in tqdm(indices):
        embedding_interactions = explainer.interactions(batch_embedding,
                                                        baseline_embedding,
                                                        batch_size=FLAGS.batch_size,
                                                        num_samples=FLAGS.num_samples,
                                                        use_expectation=False,
                                                        output_indices=0,
                                                        verbose=False,
                                                        interaction_index=interaction_index)
        interaction_matrix[:,
                           interaction_index[0],
                           interaction_index[1],
                           :,
                           :] = embedding_interactions
    np.save('interaction_matrix_{}.npy'.format(FLAGS.model_type),
            interaction_matrix)

if __name__ == '__main__':
    app.run(interpret)
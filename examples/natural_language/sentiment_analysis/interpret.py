import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from path_explain.path_explainer_tf import PathExplainerTF
from path_explain import utils

import train
from model import cnn_model
from preprocess import sentiment_dataset

from absl import app
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_integer('num_samples', 300, 'Number of samples to draw for attributions')
flags.DEFINE_integer('num_sentences', 100, 'Number of sentences to interpret')
flags.DEFINE_integer('use_custom_sentences', False, 'Set to true to interpret custom sentences')

def interpret(argv=None):
    print('Setting up environment...')
    utils.set_up_environment(visible_devices=FLAGS.visible_devices)

    print("Loading data...")
    train_set, vald_set = sentiment_dataset(batch_size=FLAGS.batch_size,
                                            max_sequence_length=FLAGS.sequence_length)
    encoder = tfds.features.text.TokenTextEncoder.load_from_file('encoder')


    print('Loading model...')
    interpret_model = cnn_model(encoder.vocab_size,
                                FLAGS.embedding_dim,
                                FLAGS.sequence_length,
                                FLAGS.dropout_rate,
                                FLAGS.num_filters,
                                FLAGS.hidden_units,
                                for_interpretation=True)

    model = tf.keras.models.load_model('model.h5')
    embedding_model = tf.keras.models.Model(model.input, model.layers[1].output)

    interpret_model.load_weights('model.h5', by_name=True)

    explainer = PathExplainerTF(interpret_model)


    if use_custom_sentences:
        custom_sentences = ['This movie was good',
                            'This movie was not good']

    num_accumulated = 0
    accumulated_inputs = []
    accumulated_embeddings = []
    for i, (batch_input, batch_label) in enumerate(vald_set):
        batch_embedding = embedding_model(batch_input)

        batch_pred = model(batch_input)
        batch_pred_max = (batch_pred[:, 0].numpy() > 0.5).astype(int)

        correct_mask = batch_pred_max == batch_label

        accumulated_inputs.append(batch_input[correct_mask])
        accumulated_embeddings.append(batch_embedding[correct_mask])
        num_accumulated += np.sum(correct_mask)
        if num_accumulated >= FLAGS.num_sentences:
            break

    accumulated_inputs = tf.concat(accumulated_inputs, axis=0)
    accumulated_embeddings = tf.concat(accumulated_embeddings, axis=0)
    np.save('accumulated_inputs.npy', accumulated_inputs.numpy())
    np.save('accumulated_embeddings.npy', accumulated_embeddings.numpy())

    baseline_input = np.zeros(accumulated_inputs[0:1].shape)
    baseline_embedding = embedding_model(baseline_input)

    print('Getting attributions...')
    # Get word-level attributions
    embedding_attributions = explainer.attributions(accumulated_embeddings,
                                                    baseline_embedding,
                                                    batch_size=FLAGS.batch_size,
                                                    num_samples=FLAGS.num_samples,
                                                    use_expectation=False,
                                                    output_indices=0,
                                                    verbose=True)
    np.save('embedding_attributions.npy', embedding_attributions)

    print('Getting interactions...')
    # Get pairwise word interactions
    max_indices = np.max(np.sum(accumulated_inputs != 0, axis=-1))
    interaction_matrix = np.zeros((accumulated_embeddings.shape[0],
                                   max_indices,
                                   FLAGS.embedding_dim,
                                   FLAGS.sequence_length,
                                   FLAGS.embedding_dim))

    indices = np.indices((max_indices,
                          FLAGS.embedding_dim))
    indices = indices.reshape(2, -1)
    indices = indices.swapaxes(0, 1)
    for interaction_index in tqdm(indices):
        embedding_interactions = explainer.interactions(accumulated_embeddings,
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
    np.save('interaction_matrix.npy', interaction_matrix)

if __name__ == '__main__':
    app.run(interpret)
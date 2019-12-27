
import tensorflow as tf
import numpy as np
from path_explain.utils import set_up_environment
from path_explain.path_explainer_tf import PathExplainerTF

from preprocess import higgs_dataset

from absl import app
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_integer('num_examples', 50000, 'Number of inputs to run attributions on')
flags.DEFINE_integer('num_samples', 200, 'Number of samples to use when computing attributions')

def interpret(argv=None):
    set_up_environment(visible_devices=FLAGS.visible_devices)

    train_set, test_set, vald_set = higgs_dataset(batch_size=FLAGS.batch_size,
                                                  num_parallel_calls=8,
                                                  buffer_size=10000,
                                                  seed=0,
                                                  scale=True,
                                                  include_vald=True)

    print('Loading model...')
    model = build_model(dropout_rate=FLAGS.dropout_rate,
                        num_layers=FLAGS.num_layers,
                        hidden_units=FLAGS.hidden_units,
                        for_interpretation=True)
    model.load_weights('model.h5', by_name=True)

    print('Gathering inputs...')
    training_iters = int(10000 / FLAGS.batch_size)
    training_samples = []
    for i, (x_batch, _) in train_set:
        training_samples.append(x_batch)
        if i > training_iters:
            break
    training_samples = tf.concat(training_samples, axis=0)

    test_iters = int(FLAGS.num_examples / FLAGS.batch_size)
    input_samples = []
    for i, (x_batch, _) in test_set:
        input_samples.append(x_batch)
        if i > test_iters:
            break
    input_samples = tf.concat(test_samples, axis=0)

    explainer = PathExplainerTF(model)
    print('Computing attributions...')
    attributions = explainer.attributions(inputs=input_samples,
                                          baseline=training_samples,
                                          batch_size=FLAGS.batch_size,
                                          num_samples=FLAGS.num_samples,
                                          use_expectation=True,
                                          output_indices=0,
                                          verbose=True)
    np.save('attributions.npy', attributions)

    print('Computing interactions...')
    interactions = explainer.interactions(inputs=input_samples,
                                          baseline=training_samples,
                                          batch_size=FLAGS.batch_size,
                                          num_samples=FLAGS.num_samples,
                                          use_expectation=True,
                                          output_indices=0,
                                          verbose=True)
    np.save('interactions.npy', interactions)

if __name__ == '__main__':
    app.run(interpret)
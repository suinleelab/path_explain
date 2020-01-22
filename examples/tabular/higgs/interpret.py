import tensorflow as tf
import numpy as np
from path_explain.utils import set_up_environment
from path_explain.path_explainer_tf import PathExplainerTF

from preprocess import higgs_dataset
from train import build_model

from absl import app
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_integer('num_examples', 10000, 'Number of inputs to run attributions on')
flags.DEFINE_integer('num_samples', 300, 'Number of samples to use when computing attributions')

def interpret(argv=None):
    set_up_environment(visible_devices=FLAGS.visible_devices)

    train_set, test_set, vald_set = higgs_dataset(batch_size=FLAGS.batch_size,
                                                  num_parallel_calls=8,
                                                  buffer_size=10000,
                                                  seed=0,
                                                  scale=True,
                                                  include_vald=True)

    print('Loading model...')
    model = build_model(weight_decay=FLAGS.weight_decay,
                        num_layers=FLAGS.num_layers,
                        hidden_units=FLAGS.hidden_units,
                        for_interpretation=True)
    model.load_weights('model.h5', by_name=True)

    print('Gathering inputs...')
    training_iters = int(10000 / FLAGS.batch_size)
    training_samples = []
    for i, (x_batch, _) in enumerate(train_set):
        training_samples.append(x_batch)
        if i >= training_iters:
            break
    training_samples = tf.concat(training_samples, axis=0)

    input_samples = []
    true_labels = []
    pred_output = []
    num_accumulated = 0
    for x_batch, label_batch in test_set:
        pred_labels = model(x_batch)
        correct_mask = (pred_labels[:, 0].numpy() > 0.5).astype(int) == label_batch

        input_samples.append(x_batch.numpy()[correct_mask])
        pred_output.append(pred_labels.numpy()[correct_mask, 0])
        true_labels.append(label_batch.numpy()[correct_mask])
        num_accumulated += np.sum(correct_mask)

        if num_accumulated >= FLAGS.num_examples:
            break

    input_samples = np.concatenate(input_samples, axis=0).astype(np.float32)
    true_labels = np.concatenate(true_labels, axis=0)
    pred_output = np.concatenate(pred_output, axis=0)

    np.save('input_samples.npy', input_samples)
    np.save('pred_output.npy', pred_output)
    np.save('true_labels.npy', true_labels)

    explainer = PathExplainerTF(model)
    print('Computing attributions...')
    attributions = explainer.attributions(inputs=input_samples,
                                          baseline=np.zeros((1, input_samples.shape[1]), dtype=np.float32),
                                          batch_size=FLAGS.batch_size,
                                          num_samples=FLAGS.num_samples,
                                          use_expectation=False,
                                          output_indices=0,
                                          verbose=True)
    np.save('attributions.npy', attributions)

    print('Computing interactions...')
    interactions = explainer.interactions(inputs=input_samples,
                                          baseline=np.zeros((1, input_samples.shape[1]), dtype=np.float32),
                                          batch_size=FLAGS.batch_size,
                                          num_samples=FLAGS.num_samples,
                                          use_expectation=False,
                                          output_indices=0,
                                          verbose=True)
    np.save('interactions.npy', interactions)

if __name__ == '__main__':
    app.run(interpret)
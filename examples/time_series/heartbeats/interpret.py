import tensorflow as tf
import numpy as np

from model import cnn_model
from preprocess import mitbih_dataset
from path_explain.path_explainer_tf import PathExplainerTF
from path_explain.utils import set_up_environment

from absl import app
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_integer('batch_size', 32, 'Batch size for interpretation')
flags.DEFINE_integer('num_examples', 100, 'Number of examples per class to explain')
flags.DEFINE_integer('num_samples',  200, 'Number of samples to draw when computing attributions')
flags.DEFINE_string('visible_devices', '0', 'Which gpu to train on')

def interpret(argv=None):
    set_up_environment(visible_devices=FLAGS.visible_devices)

    print('Reading data...')
    x_train, y_train, x_test, y_test = mitbih_dataset()
    print('Dataset shape: {}'.format(x_train.shape))
    print('Loading model...')
    original_model = tf.keras.models.load_model('model.h5')

    interpret_model = cnn_model(for_interpretation=True)
    interpret_model.load_weights('model.h5', by_name=True)

    y_pred = original_model.predict(x_test)
    y_pred_max = np.argmax(y_pred, axis=-1)

    explainer = PathExplainerTF(interpret_model)

    for c in range(5):
        print('Interpreting class {}'.format(c))
        class_mask = np.logical_and(y_test == c,
                                    y_pred_max == y_test)
        class_indices = np.where(class_mask)[0][:FLAGS.num_examples]

        batch_samples = x_test[class_indices]

        attributions = explainer.attributions(inputs=batch_samples,
                                              baseline=x_train,
                                              batch_size=FLAGS.batch_size,
                                              num_samples=FLAGS.num_samples,
                                              use_expectation=True,
                                              output_indices=c,
                                              verbose=True)
        np.save('attributions_{}.npy'.format(c), attributions)

        interactions = explainer.interactions(inputs=batch_samples,
                                              baseline=x_train,
                                              batch_size=FLAGS.batch_size,
                                              num_samples=FLAGS.num_samples,
                                              use_expectation=True,
                                              output_indices=c,
                                              verbose=True)
        np.save('interactions_{}.npy'.format(c), interactions)

if __name__ == '__main__':
    app.run(interpret)
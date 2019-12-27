import tensorflow as tf
import numpy as np
from joblib import dump, load
from sklearn.decomposition import PCA
from train import load_mnist, build_model

from path_explain.path_explainer_tf import PathExplainerTF
from path_explain.utils import set_up_environment

from absl import app
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_boolean('interpret_pca', False, 'Set to true to interpret in the PCA basis.')
flags.DEFINE_integer('max_images', 100, 'Number of images to interpret.')
flags.DEFINE_integer('num_samples', 100, 'Number of samples to use when computing attributions.')
flags.DEFINE_integer('visible_device', None, 'GPU to use.')

def interpret(argv=None):
    if FLAGS.visible_device is not None:
        set_up_environment(visible_devices=str(FLAGS.visible_device))
    else:
        set_up_environment()

    print('Loading data...')
    (x_train, y_train), (x_test, y_test) = load_mnist()

    if FLAGS.interpret_pca:
        print('Fitting PCA...')
        reshaped_x_train = np.reshape(x_train, (x_train.shape[0], -1))
        pca_model = PCA()
        pca_model.fit(reshaped_x_train)
        dump(pca_model, 'pca_model.pickle')

        transformed_x_train = pca_model.transform(reshaped_x_train).astype(np.float32)
        baseline = transformed_x_train

        reshaped_x_test = np.reshape(x_test, (x_test.shape[0], -1))
        transformed_x_test = pca_model.transform(reshaped_x_test).astype(np.float32)
    else:
        baseline = x_train
        transformed_x_test = x_test

    print('Loading model...')
    original_model = build_model(for_interpretation=True)
    original_model.load_weights('model.h5', by_name=True)

    if FLAGS.interpret_pca:
        interpret_model = tf.keras.models.Sequential()
        interpret_model.add(tf.keras.layers.Input(shape=(784)))
        pca_layer = tf.keras.layers.Dense(units=784,
                                          activation=None,
                                          use_bias=True,
                                          kernel_initializer=tf.keras.initializers.Constant(pca_model.components_),
                                          bias_initializer=tf.keras.initializers.Constant(pca_model.mean_))
        interpret_model.add(pca_layer)
        interpret_model.add(tf.keras.layers.Reshape((28, 28, 1)))
        interpret_model.add(original_model)
    else:
        interpret_model = original_model


    flag_name = ''
    if FLAGS.interpret_pca:
        flag_name += '_pca'

    print('Getting attributions...')
    explainer = PathExplainerTF(interpret_model)
    explained_inputs = transformed_x_test[:FLAGS.max_images]
    explained_labels = y_test[:FLAGS.max_images].astype(int)
    attributions = explainer.attributions(inputs=explained_inputs,
                                          baseline=baseline,
                                          batch_size=FLAGS.batch_size,
                                          num_samples=FLAGS.num_samples,
                                          use_expectation=True,
                                          output_indices=explained_labels,
                                          verbose=True)
    np.save('attributions{}.npy'.format(flag_name),
            attributions)

    print('Getting interactions...')
    interactions = explainer.interactions(inputs=explained_inputs,
                                          baseline=baseline,
                                          batch_size=FLAGS.batch_size,
                                          num_samples=FLAGS.num_samples,
                                          use_expectation=True,
                                          output_indices=explained_labels,
                                          verbose=True)
    np.save('interactions{}.npy'.format(flag_name),
            interactions)

if __name__ == '__main__':
    app.run(interpret)
import time
import tensorflow as tf
import numpy as np
import pandas as pd
from tqdm import tqdm
from path_explain import utils, PathExplainerTF

from absl import app, flags
FLAGS = flags.FLAGS

flags.DEFINE_string('model', 'vgg16',
                    'One of `vgg16`, `inception_v3`, `mobilenet_v2`')
flags.DEFINE_string('type', 'attribution',
                    'One of `attribution`, `single_interaction`, `jacobian_interactions`, `loop_interactions`')
flags.DEFINE_string('visible_devices', '0', 'GPU to use')

def time_model(model, test_image, baseline_image):
    explainer = PathExplainerTF(model)

    elapsed_time = np.nan

    if FLAGS.type == 'attribution':
        try:
            start_time = time.time()
            _ = explainer.attributions(inputs=test_image,
                                       baseline=baseline_image,
                                       batch_size=100,
                                       num_samples=200,
                                       use_expectation=False,
                                       output_indices=0)
            elapsed_time = time.time() - start_time
            print('Computing the IG saliency map took: {:.2f} seconds'.format(elapsed_time))
        except tf.errors.ResourceExhaustedError:
            print('Failed to compute the IG saliency map')
    elif FLAGS.type == 'single_interaction':
        try:
            start_time = time.time()
            _ = explainer.interactions(inputs=test_image,
                                       baseline=baseline_image,
                                       batch_size=32,
                                       num_samples=200,
                                       use_expectation=False,
                                       output_indices=0,
                                       interaction_index=(0, 0, 0))
            elapsed_time = time.time() - start_time
            print('A single double backpropagation took: {:.2f} seconds'.format(elapsed_time))
        except tf.errors.ResourceExhaustedError:
            print('Failed to double back propagate')
    elif FLAGS.type == 'jacobian_interactions':
        try:
            start_time = time.time()
            _ = explainer.interactions(inputs=test_image,
                                       baseline=baseline_image,
                                       batch_size=1,
                                       num_samples=200,
                                       use_expectation=False,
                                       output_indices=0)
            elapsed_time = time.time() - start_time
            print('Computing the IH interactions took: {:.2f} seconds'.format(elapsed_time))
        except tf.errors.ResourceExhaustedError:
            print('Failed to compute the IH interactions via the jacobian')
    elif FLAGS.type == 'loop_interactions':
        try:
            start_time = time.time()
            for i in range(test_image.shape[1]):
                for j in range(test_image.shape[2]):
                    for c in range(test_image.shape[3]):
                        interaction_index = (i, j, c)
                        _ = explainer.interactions(inputs=test_image,
                                                   baseline=baseline_image,
                                                   batch_size=100,
                                                   num_samples=200,
                                                   use_expectation=False,
                                                   output_indices=0,
                                                   interaction_index=(i, j, c))
            elapsed_time = time.time() - start_time
            print('Computing the IH interactions by looping took: {:.2f} seconds'.format(elapsed_time))
        except tf.errors.ResourceExhaustedError:
            print('Failed to compute the IH interactions by looping')
    else:
        raise ValueError('Unrecognized value `{}` for argument `type`'.format(FLAGS.type))

    return elapsed_time

def main(argv=None):
    utils.set_up_environment(visible_devices=FLAGS.visible_devices)

    model_dict = {
        'vgg16': tf.keras.applications.vgg16.VGG16,
        'inception_v3': tf.keras.applications.inception_v3.InceptionV3,
        'mobilenet_v2': tf.keras.applications.mobilenet_v2.MobileNetV2
    }
    sizes = {
        'vgg16': 224,
        'inception_v3': 299,
        'mobilenet_v2': 224
    }
    model = model_dict[FLAGS.model]()
    image_size = sizes[FLAGS.model]

    test_image = np.random.uniform(-1.0, 1.0, size=(1, image_size, image_size, 3)).astype(np.float32)
    baseline_image = np.zeros((1, image_size, image_size, 3)).astype(np.float32)
    elapsed_time = time_model(model, test_image, baseline_image)

    data_dictionary = {
        'Model': [FLAGS.model],
        'Type':  [FLAGS.type],
        'Time':  [elapsed_time]
    }

    data = pd.DataFrame(data_dictionary)
    data.to_csv('{}_{}.csv'.format(FLAGS.model, FLAGS.type), index=False)

if __name__ == '__main__':
    app.run(main)

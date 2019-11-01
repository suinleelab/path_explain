import os
import tensorflow as tf
import numpy as np
import shap
from marginal import MarginalExplainer
from colour import Color

import utils
from absl import app
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_string('dataset', 'mnist', 'One of `mnist`, `color_mnist`')
flags.DEFINE_integer('batch_size', 50, 'Batch size for training and evaluation')
flags.DEFINE_integer('num_epochs', 5, 'Number of epochs to train for')
flags.DEFINE_float('learning_rate', 0.1, 'Initial learning rate to use while training')
flags.DEFINE_string('background', 'black', 'One of `black`, `train_dist`')
flags.DEFINE_integer('num_shap_samples', 100, 'Number of test-set examples to evaluate attributions on')

def build_model(input_shape=(28, 28, 1)):
    model = tf.keras.models.Sequential()
    
    model.add(tf.keras.layers.Input(shape=input_shape, dtype=tf.float32))
    model.add(tf.keras.layers.Conv2D(32, (3, 3), padding='same'))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(tf.keras.layers.Conv2D(64, (3, 3), padding='same'))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1024))
    model.add(tf.keras.layers.Activation('relu'))

    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(10))
    model.add(tf.keras.layers.Activation('softmax'))
    return model

def get_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()    
    x_train = x_train.astype('float32')
    x_test  = x_test.astype('float32')
    
    if FLAGS.dataset == 'color_mnist':
        red = Color("red")
        colors = list(red.range_to(Color("purple"), 10))
        colors = [np.asarray(x.get_rgb()) for x in colors]

        x_train_color = np.zeros((x_train.shape[0], 28, 28, 3), dtype = np.float32)
        for i in range(x_train.shape[0]):
            my_color         = colors[y_train[i]]
            x_train_color[i] = x_train[i].astype(np.float32)[:, :, np.newaxis] * my_color[None, None, :]

        x_test_color = np.zeros((x_test.shape[0], 28, 28, 3), dtype = np.float32)
        for i in range(x_test.shape[0]):
            my_color = colors[9 - y_test[i]]
            x_test_color[i] = x_test[i].astype(np.float32)[:, :, np.newaxis] * my_color[None, None, :]
        x_train = x_train_color
        x_test  = x_test_color
    else:
        x_train = np.expand_dims(x_train, axis=-1)
        x_test  = np.expand_dims(x_test, axis=-1)
    
    x_train = x_train * (1. / 255) - 0.5
    x_test  = x_test  * (1. / 255) - 0.5
#     y_train = tf.keras.utils.to_categorical(y_train, 10)
#     y_test  = tf.keras.utils.to_categorical(y_test,  10)
    return x_train, y_train, x_test, y_test

def train(argv=None):
    input_shape = (28, 28, 1)
    if FLAGS.dataset == 'color_mnist':
        input_shape = (28, 28, 3)
        
    model = build_model(input_shape)
    x_train, y_train, x_test, y_test = get_data()
    
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)
    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
    model.fit(x_train, y_train, 
              batch_size=FLAGS.batch_size, 
              epochs=FLAGS.num_epochs, 
              validation_data=(x_test, y_test))
    
    if FLAGS.background == 'black':
        background = np.zeros((1, *input_shape))
    elif FLAGS.background == 'train_dist':
        background = x_train[:200]
    
    primal_explainer = MarginalExplainer(model, background, 
                                         nsamples=200, representation='mobius')
    primal_effects = primal_explainer.explain(x_test[:FLAGS.num_shap_samples], verbose=True)
    
    model_func = lambda x: model(x.reshape(x.shape[0], *input_shape).astype(np.float32)).numpy()
    kernel_explainer = shap.KernelExplainer(model_func, 
                                            np.reshape(background, (background.shape[0], -1)))
    shap_values = kernel_explainer.shap_values(x_test[:FLAGS.num_shap_samples].reshape(
                                               FLAGS.num_shap_samples, -1))
    shap_values = np.reshape(shap_values, (num_shap_samples, *input_shape))
    
    interaction_effects = shap_values - primal_effects
    
    os.makedirs('data/{}/'.format(FLAGS.dataset), exist_ok=True)
    np.save('data/{}/primal_effects.npy'.format(FLAGS.dataset), primal_effects)
    np.save('data/{}/shap_values.npy'.format(FLAGS.dataset),    shap_values)
    np.save('data/{}/interaction_effects.npy'.format(FLAGS.dataset), interaction_effects)
    
if __name__ == '__main__':
    utils.set_up_environment()
    app.run(train)
import os
import tensorflow as tf
import numpy as np
import shap
from interaction_effects.marginal import MarginalExplainer
from colour import Color

from interaction_effects import utils
from absl import app
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_string('dataset', 'mnist', 'One of `mnist`, `color_mnist`')
flags.DEFINE_integer('batch_size', 50, 'Batch size for training and evaluation')
flags.DEFINE_integer('num_epochs', 10, 'Number of epochs to train for')
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate to use while training')
flags.DEFINE_string('background', 'black', 'One of `black`, `train_dist`')
flags.DEFINE_integer('num_shap_samples', 10, 'Number of test-set examples to evaluate attributions on')
flags.DEFINE_boolean('train_only', False, 'Set to true to only train the model')

def build_model(input_shape=(28, 28, 1)):
    weight_decay = 0.001
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=input_shape, dtype=tf.float32))
    model.add(tf.keras.layers.Conv2D(20, (5, 5), padding='same', kernel_regularizer=tf.keras.regularizers.l2(weight_decay)))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(tf.keras.layers.Conv2D(50, (5, 5), padding='same', kernel_regularizer=tf.keras.regularizers.l2(weight_decay)))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(526, kernel_regularizer=tf.keras.regularizers.l2(weight_decay)))
    model.add(tf.keras.layers.Activation('relu'))

    model.add(tf.keras.layers.Dense(10, kernel_regularizer=tf.keras.regularizers.l2(weight_decay)))
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

def save_attributions(model, samples, labels, background, input_shape, subdir='train'):
    os.makedirs('data/{}/{}/'.format(FLAGS.dataset, subdir), exist_ok=True)

    primal_explainer = MarginalExplainer(model, background, 
                                         nsamples=200, representation='mobius')
    primal_effects = primal_explainer.explain(samples, verbose=True, index_outputs=True, labels=labels)
    
    model_func = lambda x: model(np.reshape(x, (x.shape[0], *input_shape)).astype(np.float32)).numpy()
    
    if FLAGS.background == 'train_dist':
        shap_indices = np.random.choice(background.shape[0], size=200, replace=False)
        background = background[shap_indices]
        
    sample_explainer = shap.SamplingExplainer(model_func,
                                              np.reshape(background, 
                                                         (background.shape[0], -1)))
    shap_values = sample_explainer.shap_values(np.reshape(samples, 
                                                          (FLAGS.num_shap_samples, -1)))
    shap_values = np.stack(shap_values, axis=0)
    shap_values = shap_values[labels, np.arange(shap_values.shape[1]), :]
    
#     grad_explainer = shap.GradientExplainer(model, background)
#     shap_values = grad_explainer.shap_values(samples, nsamples=200, ranked_outputs=1)
    shap_values = np.reshape(shap_values, (FLAGS.num_shap_samples, *input_shape))
    
    interaction_effects = shap_values - primal_effects
    
    np.save('data/{}/{}/primal_effects_{}.npy'.format(FLAGS.dataset,      subdir, FLAGS.background), primal_effects)
    np.save('data/{}/{}/shap_values_{}.npy'.format(FLAGS.dataset,         subdir, FLAGS.background), shap_values)
    np.save('data/{}/{}/interaction_effects_{}.npy'.format(FLAGS.dataset, subdir, FLAGS.background), interaction_effects)

def train(argv=None):
    input_shape = (28, 28, 1)
    if FLAGS.dataset == 'color_mnist':
        input_shape = (28, 28, 3)
        
    model = build_model(input_shape)
    x_train, y_train, x_test, y_test = get_data()
    
    try:
        model = tf.keras.models.load_model('models/{}_model.h5'.format(FLAGS.dataset))
    except OSError:
        optimizer = tf.keras.optimizers.Adam(learning_rate=FLAGS.learning_rate)
        model.compile(optimizer=optimizer,
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                      metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
        model.fit(x_train, y_train, 
                  batch_size=FLAGS.batch_size, 
                  epochs=FLAGS.num_epochs, 
                  validation_data=(x_test, y_test))
        model.save('models/{}_model.h5'.format(FLAGS.dataset))
    
    if FLAGS.background == 'black':
        background = np.zeros((1, *input_shape)).astype(np.float32)
    elif FLAGS.background == 'train_dist':
        background = x_train
    
    if not FLAGS.train_only:
        save_attributions(model, x_test[:FLAGS.num_shap_samples],  y_test[:FLAGS.num_shap_samples].astype(int), 
                          background, input_shape, subdir='test')
        save_attributions(model, x_train[:FLAGS.num_shap_samples], y_train[:FLAGS.num_shap_samples].astype(int),
                          background, input_shape, subdir='train')
    
    
if __name__ == '__main__':
    utils.set_up_environment()
    app.run(train)

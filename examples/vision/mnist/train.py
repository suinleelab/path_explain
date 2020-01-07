import tensorflow as tf
import numpy as np

from absl import app
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_integer('batch_size', 128, 'Batch size for training')
flags.DEFINE_integer('num_epochs', 10, 'Number of epochs to train for')
flags.DEFINE_float('learning_rate', 0.001, 'Learning rate for SGD')

def build_model(for_interpretation=False):
    activation_function = tf.keras.activations.relu
    if for_interpretation:
        activation_function = tf.keras.activations.softplus

    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Input(shape=(28, 28, 1),
                                    dtype=tf.float32,
                                    name='input'))
    model.add(tf.keras.layers.Conv2D(32, (3, 3),
                                     padding='same',
                                     activation=activation_function,
                                     name='conv1'))
    model.add(tf.keras.layers.Conv2D(64, (3, 3),
                                     padding='same',
                                     activation=activation_function,
                                     name='conv2'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2),
                                           name='pool'))

    model.add(tf.keras.layers.Dropout(0.25, name='dropout1'))
    model.add(tf.keras.layers.Flatten(name='flatten'))
    model.add(tf.keras.layers.Dense(128,
                                    activation=activation_function,
                                    name='dense'))

    model.add(tf.keras.layers.Dropout(0.5,
                                      name='dropout2'))
    model.add(tf.keras.layers.Dense(10, name='dense_out'))

    if not for_interpretation:
        model.add(tf.keras.layers.Activation('softmax', name='softmax'))

    return model

def load_mnist():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = np.expand_dims(x_train, axis=-1).astype(np.float32)
    x_test  = np.expand_dims(x_test,  axis=-1).astype(np.float32)

    x_train = x_train * (1. / 255) - 0.5
    x_test  = x_test  * (1. / 255) - 0.5

    return (x_train, y_train), (x_test, y_test)

def train(argv=None):
    (x_train, y_train), (x_test, y_test) = load_mnist()

    model = build_model()
    optimizer = tf.keras.optimizers.Adam(learning_rate=FLAGS.learning_rate)
    loss = tf.keras.losses.SparseCategoricalCrossentropy()
    metrics = [tf.keras.metrics.SparseCategoricalAccuracy()]

    model.compile(loss=loss,
                  optimizer=optimizer,
                  metrics=metrics)

    model.fit(x_train,
              y_train,
              batch_size=FLAGS.batch_size,
              epochs=FLAGS.num_epochs,
              validation_data=(x_test, y_test),
              verbose=1)

    tf.keras.models.save_model(model, 'model.h5')

if __name__ == '__main__':
    app.run(train)
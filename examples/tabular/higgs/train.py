import joblib
import tensorflow as tf
import numpy as np
from path_explain.utils import set_up_environment
from path_explain.path_explainer_tf import PathExplainerTF

from preprocess import higgs_dataset

from absl import app
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_integer('num_layers', 5, 'Number of dense layers')
flags.DEFINE_integer('hidden_units', 300, 'Number of units in each dense layer')
flags.DEFINE_integer('batch_size', 100, 'Batch size to use while training')
flags.DEFINE_integer('epochs', 200, 'Maximum number of epochs to train for')
flags.DEFINE_string('visible_devices', '0', 'Which gpu to use')
flags.DEFINE_float('weight_decay', 1e-5, 'Amount of weight decay on each weight')
flags.DEFINE_float('learning_rate', 0.05, 'Initial learning rate to use for training')
flags.DEFINE_float('decay_rate', 0.96, 'Amount to decay the learning rate every epoch')
flags.DEFINE_float('momentum', 0.9, 'Momentum to use for SGD while training')

flags.DEFINE_boolean('evaluate', False, 'Set to true to evaluate the model instead of training')
flags.DEFINE_boolean('flip_indices', False, 'Set to true to flip the indices of m_wbb and m_wwbb')

def build_model(weight_decay=1e-5,
                num_layers=5,
                hidden_units=300,
                for_interpretation=False):
    activation = tf.keras.activations.tanh

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input(shape=(28,),
                                    dtype=tf.float32,
                                    name='input'))

    for i in range(num_layers):
        model.add(tf.keras.layers.Dense(units=hidden_units,
                                        activation=activation,
                                        kernel_regularizer=tf.keras.regularizers.l2(l=weight_decay),
                                        name=f'dense_{i}'))

    model.add(tf.keras.layers.Dense(units=1,
                                    activation=None,
                                    name='dense_out'))

    if not for_interpretation:
        model.add(tf.keras.layers.Activation(tf.keras.activations.softmax,
                                             name='sigmoid'))
    return model

def train(argv=None):
    set_up_environment(visible_devices=FLAGS.visible_devices)

    train_set, test_set, vald_set = higgs_dataset(batch_size=FLAGS.batch_size,
                                                  num_parallel_calls=8,
                                                  buffer_size=10000,
                                                  seed=0,
                                                  scale=True,
                                                  include_vald=True,
                                                  flip_indices=FLAGS.flip_indices)

    if FLAGS.evaluate:
        print('Evaluating model with flip indices set to {}'.format(FLAGS.flip_indices))
        model = tf.keras.models.load_model('model.h5')
        print('---------- Train Set ----------')
        model.evaluate(train_set, verbose=2)
        print('---------- Vald Set ----------')
        model.evaluate(vald_set,  verbose=2)
        return

    model = build_model(weight_decay=FLAGS.weight_decay,
                        num_layers=FLAGS.num_layers,
                        hidden_units=FLAGS.hidden_units,
                        for_interpretation=False)

    steps_per_epoch = int(10000000 / FLAGS.batch_size)
    learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=FLAGS.learning_rate,
                                                                   decay_steps=steps_per_epoch,
                                                                   decay_rate=FLAGS.decay_rate,
                                                                   staircase=True)
    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate,
                                        momentum=FLAGS.momentum,
                                        nesterov=True)

    model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                  optimizer=optimizer,
                  metrics=[tf.keras.metrics.BinaryAccuracy(),
                           tf.keras.metrics.AUC()])

    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    history = model.fit(train_set,
                        epochs=FLAGS.epochs,
                        validation_data=vald_set,
                        callbacks=[callback])
    tf.keras.models.save_model(model, 'model.h5')
    joblib.dump(history.history, 'history.pickle')

if __name__ == '__main__':
    app.run(train)

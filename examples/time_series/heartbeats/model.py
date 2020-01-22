"""
Builds a cnn for sequence classification. Taken from
https://www.kaggle.com/coni57/model-from-arxiv-1805-00794
"""

import tensorflow as tf
from path_explain.utils import softplus_activation

def cnn_model(signal_length=187,
              num_signals=1,
              for_interpretation=False):
    activation_function = tf.keras.activations.relu
    if for_interpretation:
        activation_function = softplus_activation(beta=10.0)

    inp = tf.keras.layers.Input(shape=(signal_length, num_signals), name='input')
    C   = tf.keras.layers.Conv1D(filters=32, kernel_size=5, strides=1, name='C')(inp)

    C11 = tf.keras.layers.Conv1D(filters=32, kernel_size=5, strides=1, padding='same', name='C11')(C)
    A11 = tf.keras.layers.Activation(activation_function, name='A11')(C11)
    C12 = tf.keras.layers.Conv1D(filters=32, kernel_size=5, strides=1, padding='same', name='C12')(A11)
    S11 = tf.keras.layers.Add(name='S11')([C12, C])
    A12 = tf.keras.layers.Activation(activation_function, name='A12')(S11)
    M11 = tf.keras.layers.MaxPooling1D(pool_size=5, strides=2, name='M11')(A12)

    C21 = tf.keras.layers.Conv1D(filters=32, kernel_size=5, strides=1, padding='same', name='C21')(M11)
    A21 = tf.keras.layers.Activation(activation_function, name='A21')(C21)
    C22 = tf.keras.layers.Conv1D(filters=32, kernel_size=5, strides=1, padding='same', name='C22')(A21)
    S21 = tf.keras.layers.Add(name='S21')([C22, M11])
    A22 = tf.keras.layers.Activation(activation_function, name='A22')(S11)
    M21 = tf.keras.layers.MaxPooling1D(pool_size=5, strides=2, name='M21')(A22)

    C31 = tf.keras.layers.Conv1D(filters=32, kernel_size=5, strides=1, padding='same', name='C31')(M21)
    A31 = tf.keras.layers.Activation(activation_function, name='A31')(C31)
    C32 = tf.keras.layers.Conv1D(filters=32, kernel_size=5, strides=1, padding='same', name='C32')(A31)
    S31 = tf.keras.layers.Add(name='S31')([C32, M21])
    A32 = tf.keras.layers.Activation(activation_function, name='A32')(S31)
    M31 = tf.keras.layers.MaxPooling1D(pool_size=5, strides=2, name='M31')(A32)

    C41 = tf.keras.layers.Conv1D(filters=32, kernel_size=5, strides=1, padding='same', name='C41')(M31)
    A41 = tf.keras.layers.Activation(activation_function, name='A41')(C41)
    C42 = tf.keras.layers.Conv1D(filters=32, kernel_size=5, strides=1, padding='same', name='C42')(A41)
    S41 = tf.keras.layers.Add(name='S41')([C42, M31])
    A42 = tf.keras.layers.Activation(activation_function, name='A42')(S41)
    M41 = tf.keras.layers.MaxPooling1D(pool_size=5, strides=2, name='M41')(A42)

    C51 = tf.keras.layers.Conv1D(filters=32, kernel_size=5, strides=1, padding='same', name='C51')(M41)
    A51 = tf.keras.layers.Activation(activation_function, name='A51')(C51)
    C52 = tf.keras.layers.Conv1D(filters=32, kernel_size=5, strides=1, padding='same', name='C52')(A51)
    S51 = tf.keras.layers.Add(name='S51')([C52, M41])
    A52 = tf.keras.layers.Activation(activation_function, name='A52')(S51)
    M51 = tf.keras.layers.MaxPooling1D(pool_size=5, strides=2, name='M51')(A52)

    F1 = tf.keras.layers.Flatten(name='F1')(M51)

    D1 = tf.keras.layers.Dense(32, name='D1')(F1)
    A6 = tf.keras.layers.Activation(activation_function, name='A6')(D1)
    D2 = tf.keras.layers.Dense(32, name='D2')(A6)
    D3 = tf.keras.layers.Dense(5, name='D3')(D2)
    A7 = tf.keras.layers.Softmax(name='A7')(D3)

    output_layer = A7
    if for_interpretation:
        output_layer = D3

    model = tf.keras.models.Model(inputs=inp, outputs=output_layer)
    return model

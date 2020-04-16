import tensorflow as tf
import numpy as np

class SelectInputs(tf.keras.layers.Layer):
    def __init__(self, feature_indices, num_features, name=None):
        super(SelectInputs, self).__init__(name=name)

        if isinstance(feature_indices, int):
            feature_indices = [feature_indices]

        numpy_weight = np.zeros((num_features, len(feature_indices)), dtype=np.float32)
        numpy_weight[feature_indices, np.arange(len(feature_indices))] = 1
        self.w = tf.Variable(initial_value=numpy_weight,
                             trainable=False)

    def get_config(self):
        return {'activation': None}

    def call(self, inputs):
        return tf.matmul(inputs, self.w)

def get_subnetwork(model, input_features):
    if (isinstance(input_features), int):
        input_string = int(input_features)
    else:
        input_string = '{}_{}'.format(input_features[0], input_features[1])

    input_tensor = model.get_layer('select_{}'.format(input_string)).input
    subnetwork = tf.keras.models.Model(inputs=input_tensor,
                                       outputs=model.get_layer('output_{}'.format(input_string)).output)
    output_tensor = input_tensor
    for layer in subnetwork.layers:
        output_tensor = layer(output_tensor)

    weight_multiply_index = [layer.name.split('/')[0] for layer in \
                             model.get_layer('concat').input].index('output_{}'.format(input_string))
    final_weighting = model.get_layer('output_final').weights[0][weight_multiply_index, :]
    final_weighting = tf.expand_dims(final_weighting, axis=0)
    final_weighting = tf.expand_dims(final_weighting, axis=0)

    output_tensor = tf.keras.layers.Dense(units=1,
                                          activation=None,
                                          use_bias=False,
                                          weights=final_weighting.numpy(),
                                          trainable=False,
                                          name='subnetwork_output_final')(output_tensor)
    subnetwork = tf.keras.models.Model(inputs=input_tensor,
                                       outputs=output_tensor)
    return subnetwork

def interaction_model(num_features,
                      num_layers,
                      hidden_layer_size,
                      num_outputs=1,
                      activation_function=tf.keras.activations.relu,
                      interactions_to_ignore=None):
    input_tensor = tf.keras.layers.Input(shape=num_features,
                                         name='input')

    main_effect_outputs = []
    for feature in range(num_features):
        model_output = SelectInputs(feature_indices=feature,
                                    num_features=num_features,
                                    name='select_{}'.format(feature))(input_tensor)
        for layer in range(num_layers):
            model_output = tf.keras.layers.Dense(hidden_layer_size,
                                                 activation=activation_function,
                                                 name='dense_{}_{}'.format(layer, feature))(model_output)
        model_output = tf.keras.layers.Dense(num_outputs,
                                             activation=None,
                                             name='output_{}'.format(feature))(model_output)
        main_effect_outputs.append(model_output)

    interaction_outputs = []
    for i in range(num_features):
        for j in range(i + 1, num_features):
            if interactions_to_ignore is not None and \
                (i, j) in interactions_to_ignore:
                continue

            model_output = SelectInputs(feature_indices=[i, j],
                                        num_features=num_features,
                                        name='select_{}_{}'.format(i, j))(input_tensor)
            for layer in range(num_layers):
                model_output = tf.keras.layers.Dense(hidden_layer_size,
                                                     activation=activation_function,
                                                     name='dense_{}_{}_{}'.format(layer, i, j))(model_output)
            model_output = tf.keras.layers.Dense(num_outputs,
                                                 activation=None, name='output_{}_{}'.format(i, j))(model_output)
            interaction_outputs.append(model_output)

    concatenated_outputs = tf.keras.layers.Concatenate(name='concat')(main_effect_outputs + interaction_outputs)
    weighted_model_outputs = tf.keras.layers.Dense(num_outputs,
                                                   activation=None,
                                                   use_bias=False,
                                                   name='output_final')(concatenated_outputs)
    if num_outputs == 1:
        final_output = tf.keras.layers.Activation(tf.keras.activations.sigmoid)(weighted_model_outputs)
    else:
        final_output = tf.keras.layers.Activation(tf.keras.activations.softmax)(weighted_model_outputs)

    model = tf.keras.models.Model(inputs=input_tensor, outputs=final_output)
    return model

if __name__ == '__main__':
    model = interaction_model(num_features=3,
                              num_layers=2,
                              num_outputs=1,
                              hidden_layer_size=16,
                              activation_function=tf.keras.activations.relu,
                              interactions_to_ignore=[(0, 1)])
    print(model.summary())


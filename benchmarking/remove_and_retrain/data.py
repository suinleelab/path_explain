import os
import h5py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def _combine_features(first_feature, second_feature, type_name):
    if type_name == 'multiply':
        return first_feature * second_feature
    elif type_name == 'maximum':
        return np.maximum(first_feature, second_feature)
    elif type_name == 'minimum':
        return np.minimum(first_feature, second_feature)
    elif type_name == 'squaresum':
        return np.square(first_feature + second_feature)
    elif type_name == 'sign_modify':
        return first_feature * np.sign(second_feature)
    elif type_name == 'cossum':
        return np.cos(first_feature + second_feature)
    elif type_name == 'tanhsum':
        return np.tanh(first_feature + second_feature)
    else:
        raise ValueError('Unsupported type {}'.format(type_name))

def all_interaction_indices(n):
    return np.stack(np.triu_indices(n, k=1), axis=1)

def ablate_interactions(x, interactions, spec_df, k, using_random_draw=True):
    x = x.copy()
    new_y = np.zeros(x.shape[0])

    first_indices, second_indices = np.triu_indices(x.shape[-1], k=1)
    upper_triangular_interactions = interactions[:, first_indices, second_indices]
    interaction_max_indices = np.argsort(np.abs(upper_triangular_interactions),
                                         axis=-1)[:, ::-1][:, :k]
    interaction_pairs = [[(first_indices[i], second_indices[i]) for i in select_index] \
                         for select_index in interaction_max_indices]

    mean_abs_coeff = np.max(np.abs(spec_df['coefficient']))

    for index, row in spec_df.iterrows():
        coeff = row['coefficient']
        type_name = row['interaction_type']
        i  = row['first_feature']
        j  = row['second_feature']
        pair = (i, j)

        first_feature  = x[:, pair[0]]
        second_feature = x[:, pair[1]]

        # This is the key to our remove and retrain experiments.
        # Since the interactions are known, we can regenerate the data
        # and ablate known interactions by redrawing from the distribution
        mask_use_random = [pair in sample_pairs for sample_pairs in interaction_pairs]
        mask_use_random = np.array(mask_use_random)

        if using_random_draw:
            interaction_value = _combine_features(first_feature,
                                                  second_feature,
                                                  type_name)
            random_draw = np.random.randn(x.shape[0])
            interaction_value[mask_use_random] = interaction_value[mask_use_random] * random_draw[mask_use_random]
            coeff = mean_abs_coeff
        else:
            first_feature[mask_use_random]  = np.random.randn(np.sum(mask_use_random))
            second_feature[mask_use_random] = np.random.randn(np.sum(mask_use_random))

            interaction_value = _combine_features(first_feature,
                                                  second_feature,
                                                  type_name)

        new_y += coeff * interaction_value
    return new_y

def generate_data(num_samples,
                  num_features,
                  num_interactions,
                  dataset_name,
                  interaction_types=['multiply', 'maximum', 'minimum', 'squaresum'],
                  equal_coeffs=False):
    x = np.random.randn(num_samples, num_features).astype(np.float32)

    all_possible_indices = all_interaction_indices(num_features)
    choice_indices = np.random.choice(all_possible_indices.shape[0],
                                      size=num_interactions,
                                      replace=False)
    chosen_interaction_pairs = all_possible_indices[choice_indices]
    chosen_types = np.random.choice(len(interaction_types),
                                    size=num_interactions,
                                    replace=True)
    chosen_type_names = [interaction_types[i] for i in chosen_types]
    if equal_coeffs:
        relative_interaction_coefficients = np.full(shape=num_interactions,
                                                    fill_value=1.0)
    else:
        relative_interaction_coefficients = np.random.randn(num_interactions)

    y = np.zeros(num_samples)
    for i, pair in enumerate(chosen_interaction_pairs):
        coeff = relative_interaction_coefficients[i]
        type_name = chosen_type_names[i]

        first_feature  = x[:, pair[0]]
        second_feature = x[:, pair[1]]

        interaction_value = _combine_features(first_feature,
                                              second_feature,
                                              type_name)
        y += coeff * interaction_value

    label_deviation = np.std(y)
    relative_interaction_coefficients /= label_deviation
    y /= label_deviation

    df = pd.DataFrame({
        'first_feature': chosen_interaction_pairs[:, 0],
        'second_feature': chosen_interaction_pairs[:, 1],
        'interaction_type': chosen_type_names,
        'coefficient': relative_interaction_coefficients
    })

    df.to_csv('data/{}_spec.csv'.format(dataset_name),
              index=False)


    x_train, x_test, y_train, y_test = train_test_split(x,
                                                        y,
                                                        test_size=0.2,
                                                        random_state=0)
    np.savez('data/{}.npz'.format(dataset_name),
             x_train=x_train,
             x_test=x_test,
             y_train=y_train,
             y_test=y_test)

def main(argv=None):
    num_samples = 5000
    num_features = 10
    num_interactions = 20

    generate_data(num_samples=num_samples,
                  num_features=num_features,
                  num_interactions=num_interactions,
                  dataset_name='simulated_multiply',
                  interaction_types=['multiply'])

    generate_data(num_samples=num_samples,
                  num_features=num_features,
                  num_interactions=num_interactions,
                  dataset_name='simulated_maximum',
                  interaction_types=['maximum'])

    generate_data(num_samples=num_samples,
                  num_features=num_features,
                  num_interactions=num_interactions,
                  dataset_name='simulated_minimum',
                  interaction_types=['minimum'])

    generate_data(num_samples=num_samples,
                  num_features=num_features,
                  num_interactions=num_interactions,
                  dataset_name='simulated_cossum',
                  interaction_types=['cossum'])

    generate_data(num_samples=num_samples,
                  num_features=num_features,
                  num_interactions=num_interactions,
                  dataset_name='simulated_tanhsum',
                  interaction_types=['tanhsum'])


if __name__ == '__main__':
    main()

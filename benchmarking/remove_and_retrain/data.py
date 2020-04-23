import os
import h5py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def all_interaction_indices(n):
    return np.stack(np.triu_indices(n, k=1), axis=1)

def generate_data(num_samples,
                  num_features,
                  num_interactions,
                  dataset_name,
                  interaction_types=['multiply', 'maximum', 'minimum', 'squaresum']):
    x = np.random.randn(num_samples, num_features)

    all_possible_indices = all_interaction_indices(num_features)
    choice_indices = np.random.choice(all_possible_indices.shape[0],
                                      size=num_interactions,
                                      replace=False)
    chosen_interaction_pairs = all_possible_indices[choice_indices]
    chosen_types = np.random.choice(len(interaction_types),
                                    size=num_interactions,
                                    replace=True)
    chosen_type_names = [interaction_types[i] for i in chosen_types]
    relative_interaction_coefficients = np.random.randn(num_interactions)

    y = np.zeros(num_samples)
    for i, pair in enumerate(chosen_interaction_pairs):
        coeff = relative_interaction_coefficients[i]
        type_name = chosen_type_names[i]

        first_feature  = x[:, pair[0]]
        second_feature = x[:, pair[1]]

        if type_name == 'multiply':
            y += coeff * first_feature * second_feature
        elif type_name == 'maximum':
            y += coeff * np.maximum(first_feature, second_feature)
        elif type_name == 'minimum':
            y += coeff * np.minimum(first_feature, second_feature)
        elif type_name == 'squaresum':
            y += coeff * np.square(first_feature + second_feature)
        else:
            raise ValueError('Unsupported type {}'.format(type_name))

    df = pd.DataFrame({
        'feature_pairs': chosen_interaction_pairs,
        'interaction_type': chosen_types,
        'coefficient': relative_interaction_coefficients
    })

    df.to_csv('data/{}_spec.csv'.format(dataset_name),
              index=false)

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
    generate_data(num_samples=4000,
                  num_features=10,
                  num_interactions=20,
                  dataset_name='simulated_multiply',
                  interaction_types=['multiply'])

    generate_data(num_samples=4000,
                  num_features=10,
                  num_interactions=20,
                  dataset_name='simulated_maximum',
                  interaction_types=['maximum'])

    generate_data(num_samples=4000,
                  num_features=10,
                  num_interactions=20,
                  dataset_name='simulated_minimum',
                  interaction_types=['minimum'])

    generate_data(num_samples=4000,
                  num_features=10,
                  num_interactions=20,
                  dataset_name='simulated_squaresum',
                  interaction_types=['squaresum'])

    generate_data(num_samples=4000,
                  num_features=10,
                  num_interactions=20,
                  dataset_name='simulated_all',
                  interaction_types=['multiply', 'maximum', 'minimum', 'squaresum'])

if __name__ == '__main__':
    main()

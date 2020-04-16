import tensorflow as tf
import numpy as np
from tqdm import tqdm

from sklearn.model_selection import StratifiedKFold, ParameterSampler
from build_model import interaction_model

def compile_model(model, learning_rate=0.001):
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss = tf.keras.losses.BinaryCrossentropy()
#     metrics = [tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.AUC()]
    metrics = [tf.keras.metrics.AUC()]
    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=metrics)

def get_performance(x,
                    y,
                    random_seed=0,
                    n_iter=50,
                    interactions_to_ignore=None,
                    interaction_function=None):
    np.random.seed(random_seed)
    hyper_params = {
        'learning_rate': np.exp(np.linspace(-1, -5, num=100)),
        'epochs': np.arange(25, 100),
        'num_layers': [2, 3],
        'hidden_layer_size': np.arange(8, 32)
    }
    hyper_params = list(ParameterSampler(hyper_params,
                        n_iter=n_iter,
                        random_state=random_seed))

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed)

    test_aucs = []
    split = 0

    interactions = []
    for train_index, test_index in tqdm(skf.split(x, y)):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        best_param = get_best_performing_model(hyper_params,
                                               x_train,
                                               y_train,
                                               random_seed=random_seed,
                                               interactions_to_ignore=interactions_to_ignore)

        model = interaction_model(num_features=x_train.shape[1],
                                  num_layers=best_param['num_layers'],
                                  hidden_layer_size=best_param['hidden_layer_size'],
                                  interactions_to_ignore=interactions_to_ignore)
        compile_model(model, learning_rate=best_param['learning_rate'])
        model.fit(x=x_train,
                  y=y_train,
                  batch_size=128,
                  epochs=best_param['epochs'],
                  verbose=0)
        _, test_auc = model.evaluate(x=x_test,
                                     y=y_test,
                                     batch_size=128,
                                     verbose=0)

        if interaction_function is not None:
            split_interactions = interaction_function(model, x_test, baseline=x_train)
            interactions.append(split_interactions)

        test_aucs.append(test_auc)
        split += 1
    mean_test_auc = np.mean(test_aucs)
    sd_test_auc   = np.std(test_aucs)

    print('Finished cross validation ({:.3f} +- {:.3f})'.format(mean_test_auc, sd_test_auc))

    if interaction_function is not None:
        interactions = np.concatenate(interactions, axis=0)
        return mean_test_auc, sd_test_auc, interactions
    else:
        return mean_test_auc, sd_test_auc

def get_best_performing_model(hyper_params, x, y, random_seed=0, interactions_to_ignore=None):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed)

    # Evaluating AUC, so random performance is 0.5
    best_auc = 0.5
    best_param = None

    for param in hyper_params:
        model = interaction_model(num_features=x.shape[1],
                                  num_layers=param['num_layers'],
                                  hidden_layer_size=param['hidden_layer_size'],
                                  interactions_to_ignore=interactions_to_ignore)
        compile_model(model, learning_rate=param['learning_rate'])
        model.save_weights('/tmp/temp_weights.h5')

        mean_val_auc = []
        for train_index, val_index in skf.split(x, y):
            x_train, x_val = x[train_index], x[val_index]
            y_train, y_val = y[train_index], y[val_index]

            model.load_weights('/tmp/temp_weights.h5')
            model.fit(x=x_train,
                      y=y_train,
                      batch_size=128,
                      epochs=param['epochs'],
                      verbose=0)

            _, val_auc = model.evaluate(x=x_val,
                                  y=y_val,
                                  batch_size=128,
                                  verbose=0)
            mean_val_auc.append(val_auc)
        mean_val_auc = np.mean(mean_val_auc)
        del model

        if mean_val_auc > best_auc:
            best_auc = mean_val_auc
            best_param = param

    return best_param

if __name__ == '__main__':
    x = np.random.randn(100, 5)
    y = (np.sum(x, axis=1) > 0.0).astype(int)

    mean_auc, sd_auc = get_performance(x, y, n_iter=5)

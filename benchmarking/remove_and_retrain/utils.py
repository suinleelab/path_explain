import tensorflow as tf
import numpy as np
from tqdm import tqdm

from sklearn.model_selection import StratifiedKFold, ParameterSampler, train_test_split
from build_model import interaction_model

def compile_model(model, learning_rate=0.001, regression=False):
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    if regression:
        loss = tf.keras.losses.MeanSquaredError()
        metrics = [tf.keras.metrics.MeanSquaredError()]
    else:
        loss = tf.keras.losses.BinaryCrossentropy()
        metrics = [tf.keras.metrics.AUC()]

    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=metrics)

def get_interactions(x,
                     y,
                     model,
                     interaction_function,
                     random_seed=0):
    if len(np.unique(y)) < 4:
        stratify = y
    else:
        stratify = None

    np.random.seed(random_seed)
    x_train, x_test, y_train, y_test = train_test_split(x,
                                                        y,
                                                        test_size=0.2,
                                                        random_state=random_seed,
                                                        stratify=stratify)
    interactions = interaction_function(model, x_test, baseline=x_train)
    return interactions

def get_interaction_model(x,
                          y,
                          random_seed=0):
    if len(np.unique(y)) < 3:
        regression = False
        stratify = y
    else:
        regression = True
        stratify = None

    x_train, x_test, y_train, y_test = train_test_split(x,
                                                        y,
                                                        test_size=0.2,
                                                        random_state=random_seed,
                                                        stratify=stratify)

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input(shape=(x.shape[1],)))
    model.add(tf.keras.layers.Dense(units=32, activation=tf.keras.activations.relu))
    model.add(tf.keras.layers.Dense(units=32, activation=tf.keras.activations.relu))
    model.add(tf.keras.layers.Dense(units=1,  activation=None))

    if not regression:
        model.add(tf.keras.layers.Activation(tf.keras.activations.sigmoid))

    learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.1,
                                                                   decay_steps=1,
                                                                   decay_rate=0.99,
                                                                   staircase=True)
    compile_model(model, regression=regression, learning_rate=learning_rate)
    model.fit(x_train, y_train, batch_size=300, epochs=50, verbose=1)
    model.evaluate(x_test, y_test, batch_size=300, verbose=2)
    return model

def get_performance(x,
                    y,
                    random_seed=0,
                    n_iter=50,
                    interactions_to_ignore=None,
                    best_param=None):
    if len(np.unique(y)) < 4:
        regression = False
        stratify = y
    else:
        regression = True
        stratify = None

    np.random.seed(random_seed)
    x_train, x_test, y_train, y_test = train_test_split(x,
                                                        y,
                                                        test_size=0.2,
                                                        random_state=random_seed,
                                                        stratify=stratify)
    if best_param is None:
        hyper_params = {
            'learning_rate': np.exp(np.linspace(-1, -5, num=100)),
            'epochs': np.arange(5, 20),
            'num_layers': [2, 3],
            'hidden_layer_size': np.arange(2, 6)
        }
        hyper_params = list(ParameterSampler(hyper_params,
                            n_iter=n_iter,
                            random_state=random_seed))
        best_param = get_best_performing_model(hyper_params,
                                               x_train,
                                               y_train,
                                               random_seed=random_seed,
                                               interactions_to_ignore=interactions_to_ignore)
        model = interaction_model(num_features=x_train.shape[1],
                                  num_layers=best_param['num_layers'],
                                  hidden_layer_size=best_param['hidden_layer_size'],
                                  interactions_to_ignore=interactions_to_ignore)
        compile_model(model,
                      learning_rate=best_param['learning_rate'],
                      regression=regression)
        model.fit(x=x_train,
                  y=y_train,
                  batch_size=128,
                  epochs=best_param['epochs'],
                  verbose=0)
        return best_param, model

    test_aucs = []
    for _ in range(5):
        model = interaction_model(num_features=x_train.shape[1],
                                  num_layers=best_param['num_layers'],
                                  hidden_layer_size=best_param['hidden_layer_size'],
                                  interactions_to_ignore=interactions_to_ignore)
        compile_model(model,
                      learning_rate=best_param['learning_rate'],
                      regression=regression)
        model.fit(x=x_train,
                  y=y_train,
                  batch_size=128,
                  epochs=best_param['epochs'],
                  verbose=0)
        _, test_auc = model.evaluate(x=x_test,
                                     y=y_test,
                                     batch_size=128,
                                     verbose=0)
        test_aucs.append(test_auc)
        del model

    mean_test_auc = np.mean(test_aucs)
    sd_test_auc = np.std(test_aucs)
    print('Finished training ({:.3f} +- {:.3f})'.format(mean_test_auc, sd_test_auc))

    return mean_test_auc, sd_test_auc

def get_best_performing_model(hyper_params, x, y, random_seed=0, interactions_to_ignore=None):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed)

    # Evaluating AUC, so random performance is 0.5
    best_auc = 0.5
    best_param = None
    weight_key = x.shape[0]

    for param in tqdm(hyper_params):
        model = interaction_model(num_features=x.shape[1],
                                  num_layers=param['num_layers'],
                                  hidden_layer_size=param['hidden_layer_size'],
                                  interactions_to_ignore=interactions_to_ignore)
        compile_model(model,
                      learning_rate=param['learning_rate'],
                      regression=regression)
        model.save_weights('/tmp/{}.h5'.format(weight_key))

        mean_val_auc = []
        for train_index, val_index in skf.split(x, y):
            x_train, x_val = x[train_index], x[val_index]
            y_train, y_val = y[train_index], y[val_index]

            model.load_weights('/tmp/{}.h5'.format(weight_key))
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

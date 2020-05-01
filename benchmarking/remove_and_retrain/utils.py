import tensorflow as tf
import numpy as np
from tqdm import tqdm
from data import ablate_interactions

def compile_model(model, learning_rate=0.005):
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    loss = tf.keras.losses.MeanSquaredError()
    metrics = [tf.keras.metrics.MeanSquaredError()]
    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=metrics)

def get_interactions(x_train, x_test,
                     model,
                     interaction_function):
    interactions = interaction_function(model, x_test, baseline=x_train)

def get_default_model(num_features):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input(shape=(num_features,)))
    model.add(tf.keras.layers.Dense(units=64, activation=tf.keras.activations.relu))
    model.add(tf.keras.layers.Dense(units=64, activation=tf.keras.activations.relu))
    model.add(tf.keras.layers.Dense(units=64, activation=tf.keras.activations.relu))
    model.add(tf.keras.layers.Dense(units=1,  activation=None))
    return model

def get_performance(x_train,
                    x_test,
                    model,
                    random_weights,
                    spec_df,
                    interactions_train,
                    interactions_test,
                    k=0,
                    num_iters=25,
                    batch_size=128,
                    epochs=200,
                    use_random_draw=False):

    test_performances = []
    for _ in tqdm(range(num_iters)):
        y_train = ablate_interactions(x_train,
                                      interactions_train,
                                      spec_df,
                                      k,
                                      using_random_draw=use_random_draw)
        y_test  = ablate_interactions(x_test,
                                      interactions_test,
                                      spec_df,
                                      k,
                                      using_random_draw=use_random_draw)
        model.set_weights(random_weights)

        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=5,
                                                    mode='min')
        model.fit(x=x_train,
                  y=y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=0,
                  validation_split=0.2,
                  callbacks=[callback])
        _, test_perf = model.evaluate(x=x_test,
                                      y=y_test,
                                      batch_size=batch_size,
                                      verbose=0)
        test_performances.append(test_perf)

    mean_test_performance = np.mean(test_performances)
    sd_test_performance = np.std(test_performances)
    print('Finished training ({:.3f} +- {:.3f})'.format(mean_test_performance, sd_test_performance))

    return mean_test_performance, sd_test_performance

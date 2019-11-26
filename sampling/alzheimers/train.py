import data

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import joblib
import tensorflow as tf
import numpy as np
import pandas as pd
import scipy
import altair as alt
import matplotlib.pyplot as plt
import shap
from tqdm import tqdm
from sklearn.decomposition import PCA

from interaction_effects.marginal import MarginalExplainer
from interaction_effects.plot import summary_plot
from interaction_effects import utils

from absl import app
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_integer('index', 0, 'An integer between 0 and 9')
flags.DEFINE_boolean('train_only', False, 'Set to True to train only')

def build_model(learning_rate, input_shape):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(500, input_dim=input_shape, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(250, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(100, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(10, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                  optimizer=tf.keras.optimizers.RMSprop(learning_rate=learning_rate),
                  metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.AUC()])
    return model
    
def train(argv=None):
    print('Reading data...')
    X_train_total, y_train_total, \
    X_train, y_train, \
    X_vald,  y_vald, \
    X_test,  y_test = data.load_data()
    
    learning_rates = [0.000001, 0.000005, 0.00001, 0.00005, 0.0001, 0.0005, 0.001]
    num_epochs = 500
    num_components = 500
    batch_size = 128
    
    try:
        model = tf.keras.models.load_model('model.h5')
        pca_model = joblib.load('pca.model')
        
        X_train_total_pca = pca_model.transform(X_train_total.values)
        X_test_pca = pca_model.transform(X_test.values)
        
        print('Restored model from saved checkpoint')
    except (OSError, FileNotFoundError):
        vald_aucs = []
        print('No saved model found. Training from scratch...')
        print('Finding optimal learning rate...')
        for learning_rate in learning_rates:
            model = build_model(learning_rate, num_components)
            pca_model = PCA(n_components=num_components)
            X_train_pca = pca_model.fit_transform(X_train.values)
            X_vald_pca  = pca_model.transform(X_vald.values)
            
            model.fit(X_train_pca, y_train, epochs=num_epochs, batch_size=128, verbose=0)
            score = model.evaluate(X_train_pca, y_train, batch_size=128, verbose=0)
            print('Learning rate: {}, Train Loss: {:.4f}, Train Accuracy: {:.4f}, Train AUC: {:.4f}'.format(learning_rate, score[0], score[1], score[2]))
            score = model.evaluate(X_vald_pca,  y_vald, batch_size=128, verbose=0)
            print('Vald Loss: {:.4f}, Vald Accuracy: {:.4f}, Vald AUC: {:.4f}'.format(score[0], score[1], score[2]))
            vald_aucs.append(score[2])

        print('Training Model...')
        best_auc_index = np.argmax(vald_aucs)
        print('Best learning rate was: {}'.format(learning_rates[best_auc_index]))
        model = build_model(learning_rates[best_auc_index], num_components)
        pca_model = PCA(n_components=num_components)
        X_train_total_pca = pca_model.fit_transform(X_train_total.values)
        X_test_pca = pca_model.transform(X_test.values)
            
        model.fit(X_train_total_pca, y_train_total, epochs=num_epochs, batch_size=128, verbose=0)        
        model.save('model.h5')
        joblib.dump(pca_model, 'pca.model')

    score = model.evaluate(X_test_pca, y_test, batch_size=128, verbose=0)
    print('Test Loss: {:.4f}, Test Accuracy: {:.4f}, Test AUC: {:.4f}'.format(score[0], score[1], score[2]))

    if not FLAGS.train_only:
        lower_bound = FLAGS.index * 10
        upper_bound = lower_bound + 10
        print('Getting shap values...')
        try:
            sample_shap = np.load('sample_shap{}.npy'.format(FLAGS.index))
        except FileNotFoundError:
            model_func = lambda x: model(x).numpy()
            sample_explainer = shap.SamplingExplainer(model_func, X_train_total_pca)
            sample_shap      = sample_explainer.shap_values(X_test_pca[lower_bound:upper_bound])
            np.save('sample_shap{}.npy'.format(FLAGS.index), sample_shap)

        print('Getting primal effects...')
        try:
            primal_effects = np.load('primal_effects{}.npy'.format(FLAGS.index))
        except FileNotFoundError:
            primal_explainer = MarginalExplainer(model, X_train_total_pca, X_train_total_pca.shape[0])
            primal_effects   = primal_explainer.explain(X_test_pca[lower_bound:upper_bound], batch_size=128, verbose=True)
            np.save('primal_effects{}.npy'.format(FLAGS.index), primal_effects)
        print('Done!')
    
if __name__ == '__main__':
    utils.set_up_environment()
    app.run(train)
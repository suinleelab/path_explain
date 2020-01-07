"""
A module for loading signal data. Taken from
https://www.kaggle.com/coni57/model-from-arxiv-1805-00794
"""

import pandas as pd
import numpy as np

from scipy.signal import resample
from sklearn.utils import shuffle

def stretch(x):
    l = int(187 * (1 + (np.random.uniform() - 0.5) / 3))
    y = resample(x, l)
    if l < 187:
        y_ = np.zeros(shape=(187, ))
        y_[:l] = y
    else:
        y_ = y[:187]
    return y_

def amplify(x):
    alpha  = np.random.uniform() - 0.5
    factor = -alpha * x + (1 + alpha)
    return x * factor

def augment(x):
    result = np.zeros(shape=(4, 187))
    for i in range(3):
        random_choice = np.random.uniform()

        if random_choice < 0.33:
            new_y = stretch(x)
        elif random_choice < 0.66:
            new_y = amplify(x)
        else:
            new_y = stretch(x)
            new_y = amplify(new_y)
        result[i, :] = new_y
    return result

def mitbih_dataset():
    # We re-split the data ourselves because of the huge class imbalance
    # in the test set. We re-split to have a balanced test set.
    train_df = pd.read_csv('mitbih_train.csv', header=None)
    test_df  = pd.read_csv('mitbih_test.csv',  header=None)
    df = pd.concat([train_df, test_df], axis=0)

    M = df.values
    X = M[:, :-1]
    y = M[:, -1].astype(np.int64)

    train_mat = train_df.values
    test_mat  = test_df.values

    C3 = np.argwhere(y == 3).flatten()
    result = np.apply_along_axis(augment, axis=1, arr=X[C3]).reshape(-1, 187)
    classe = np.ones(shape=(result.shape[0],), dtype=int) * 3
    X = np.vstack([X, result])
    y = np.hstack([y, classe])

    C0 = np.argwhere(y == 0).flatten()
    C1 = np.argwhere(y == 1).flatten()
    C2 = np.argwhere(y == 2).flatten()
    C3 = np.argwhere(y == 3).flatten()
    C4 = np.argwhere(y == 4).flatten()

    subC0 = np.random.choice(C0, 800)
    subC1 = np.random.choice(C1, 800)
    subC2 = np.random.choice(C2, 800)
    subC3 = np.random.choice(C3, 800)
    subC4 = np.random.choice(C4, 800)

    x_test = np.vstack([X[subC0], X[subC1], X[subC2], X[subC3], X[subC4]])
    y_test = np.hstack([y[subC0], y[subC1], y[subC2], y[subC3], y[subC4]])

    x_train = np.delete(X, [subC0, subC1, subC2, subC3, subC4], axis=0)
    y_train = np.delete(y, [subC0, subC1, subC2, subC3, subC4], axis=0)

    x_train, y_train = shuffle(x_train, y_train, random_state=0)
    x_test, y_test   = shuffle(x_test, y_test, random_state=0)

    x_train = np.expand_dims(x_train, 2).astype(np.float32)
    x_test  = np.expand_dims(x_test,  2).astype(np.float32)

    return x_train, y_train, x_test, y_test

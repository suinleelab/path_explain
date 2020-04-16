import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def heart_dataset(dir='heart.csv'):
    df = pd.read_csv(dir)

    # Filter out values for number of major (calcified) vessels
    # 4 is a placeholder for NaN in this dataset for some reason.
    # This dataset is literally riddled with inconsistencies
    # and this is one of them.
    df = df[df['ca'] != 4].copy()

    chest_pain = pd.get_dummies(df['cp'], prefix='cp')
    df = pd.concat([df, chest_pain], axis=1)
    df.drop(['cp'], axis=1, inplace=True)

    sp = pd.get_dummies(df['slope'], prefix='slope')
    th = pd.get_dummies(df['thal'],  prefix='thal')
    rest_ecg = pd.get_dummies(df['restecg'], prefix='restecg')

    frames = [df, sp, th, rest_ecg]
    df = pd.concat(frames, axis=1)
    df.drop(['slope', 'thal', 'restecg'], axis=1, inplace=True)

    X = df.drop(['target'], axis = 1)
    y = df.target.values

    # This inverts the labels. For some reason the data on kaggle
    # has the labels reversed, so we reverse them back here.
    # 1 should be patients with coronary artery disease,
    # 0 are patients without coronary artery disease
    y = (y == 0).astype(int)

    feature_names = list(X.columns)
    x_train_un, x_test_un, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2, stratify=y)

    sc = StandardScaler()

    x_train = sc.fit_transform(x_train_un)
    x_test  = sc.transform(x_test_un)

    x_train = x_train.astype(np.float32)
    x_test  = x_test.astype(np.float32)

    return x_train, y_train, x_test, y_test, feature_names, x_train_un, x_test_un

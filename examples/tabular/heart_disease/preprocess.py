import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def heart_dataset():
    df = pd.read_csv('heart.csv')

    chest_pain = pd.get_dummies(df['cp'], prefix='cp', drop_first=True)
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

    feature_names = list(X.columns)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

    sc = StandardScaler()

    x_train = sc.fit_transform(x_train)
    x_test  = sc.transform(x_test)

    x_train = x_train.astype(np.float32)
    x_test  = x_test.astype(np.float32)

    return x_train, y_train, x_test, y_test, feature_names
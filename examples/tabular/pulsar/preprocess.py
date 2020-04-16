import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def pulsar_dataset(dir='pulsar_stars.csv'):
    df = pd.read_csv(dir)

    X = df.drop(['target_class'], axis = 1)
    y = df['target_class'].values

    feature_names = list(X.columns)
    x_train_un, x_test_un, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

    sc = StandardScaler()

    x_train = sc.fit_transform(x_train_un)
    x_test  = sc.transform(x_test_un)

    x_train = x_train.astype(np.float32)
    x_test  = x_test.astype(np.float32)

    return x_train, y_train, x_test, y_test, feature_names, x_train_un, x_test_un
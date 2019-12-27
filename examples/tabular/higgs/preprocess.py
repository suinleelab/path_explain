import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from joblib import dump, load

from sklearn.preprocessing import StandardScaler

def _compute_scaling():
    batch_size = 10000
    num_total  = 10500000
    train_set, _ = higgs_dataset(batch_size=batch_size,
                                 scale=False)
    scaler = StandardScaler()

    count = 0
    for batch_input, batch_label in train_set:
        count += batch_size
        scaler.partial_fit(batch_input.numpy())
        print('{}/{}'.format(count, num_total), end='\r')
    dump(scaler, 'scaler.pickle')
    return scaler

def _read_item(item,
               scaler=None):
    label = item['class_label']
    label = tf.cast(label, tf.int64)
    del item['class_label']

    features = list(item.values())
    features = tf.stack(features, axis=0)
    features = tf.cast(features, tf.float32)

    if scaler is not None:
        features = features - scaler.mean_
        features = features / scaler.var_

    return features, label

def higgs_dataset(batch_size=1000,
                  num_parallel_calls=8,
                  buffer_size=10000,
                  seed=0,
                  scale=True,
                  include_vald=True):
    if scale:
        try:
            scaler = load('scaler.pickle')
        except FileNotFoundError:
            scaler = _compute_scaling()
    else:
        scaler=None

    dataset = tfds.load(name='higgs', split='train')
    dataset = dataset.map(lambda x: _read_item(x, scaler=scaler),
                          num_parallel_calls=num_parallel_calls)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    if include_vald:
        train_set = dataset.take(10000000)
        vald_set  = dataset.skip(10000000)
        vald_set  = dataset.take(500000)
    else:
        train_set = dataset.take(10500000)

    test_set  = dataset.skip(10500000)

    train_set = train_set.shuffle(buffer_size=buffer_size,
                                  seed=seed)

    train_set = train_set.batch(batch_size)
    test_set  = test_set.batch(batch_size)

    if include_vald:
        vald_set = vald_set.batch(batch_size)
        dataset_tuple = (train_set, test_set, vald_set)
    else:
        dataset_tuple = (train_set, test_set)

    return dataset_tuple

if __name__ == '__main__':
    _compute_scaling()
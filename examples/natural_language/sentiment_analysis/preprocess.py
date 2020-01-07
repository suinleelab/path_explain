"""
Loads the tensorflow_datasets
glue/sst2 dataset in model-readable form.
Mostly taken from the loading text tutorial:
https://www.tensorflow.org/tutorials/load_data/text
"""

import tensorflow as tf
import tensorflow_datasets as tfds

def _read_example(example):
    sentence = example['sentence']
    label = example['label']
    return sentence, label

def sentiment_dataset(batch_size=64,
                      max_sequence_length=52,
                      num_parallel_calls=8,
                      buffer_size=1000,
                      seed=0):

    data = tfds.load('glue/sst2')
    train_set = data['train']
    vald_set  = data['validation']

    try:
        encoder = tfds.features.text.TokenTextEncoder.load_from_file('encoder')
    except tf.errors.NotFoundError:
        tokenizer = tfds.features.text.Tokenizer()

        vocabulary_set = set()
        for example in train_set:
            example_tokens = tokenizer.tokenize(train_set.numpy())
            vocabulary_set.update(example_tokens)

        encoder = tfds.features.text.TokenTextEncoder(vocabulary_set)
        encoder.save_to_file('encoder')


    train_set = train_set.map(_read_example, num_parallel_calls=num_parallel_calls)
    vald_set  = vald_set.map(_read_example,  num_parallel_calls=num_parallel_calls)

    def encode(sentence, label):
        index_tensor = encoder.encode(sentence.numpy())
        return index_tensor, label

    def encode_map_fn(sentence, label):
        return tf.py_function(encode, inp=[sentence, label], Tout=(tf.int64, tf.int64))

    train_set = train_set.map(encode_map_fn, num_parallel_calls=num_parallel_calls)
    vald_set  = vald_set.map(encode_map_fn,  num_parallel_calls=num_parallel_calls)

    train_set = train_set.shuffle(buffer_size=buffer_size,
                                  seed=seed)

    train_set = train_set.padded_batch(batch_size, padded_shapes=([max_sequence_length], []))
    vald_set  = vald_set.padded_batch(batch_size,  padded_shapes=([max_sequence_length], []))

    return train_set, vald_set
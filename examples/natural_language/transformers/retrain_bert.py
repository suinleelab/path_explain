"""
A module to re-train BERT on various GLUE tasks.
This file was almost entirely ripped from the huggingface transformers repository:
https://github.com/huggingface/transformers/blob/master/examples/run_tf_glue.py
"""

import os
import tensorflow as tf
import tensorflow_datasets
from transformers import BertTokenizer, TFBertForSequenceClassification, \
                         BertConfig, glue_convert_examples_to_features, \
                         glue_processors
from absl import app
from absl import flags
from path_explain import utils

FLAGS = flags.FLAGS
flags.DEFINE_integer('batch_size', 32, 'Batch size for training')
flags.DEFINE_float('learning_rate', 3e-5, 'Learning rate for training')
flags.DEFINE_float('epsilon', 1e-8, 'Epsilon to use for ADAM optimization')
flags.DEFINE_integer('hidden_size', 60, 'Size of the embedding layer')
flags.DEFINE_integer('max_length', 128, 'The maximum length of any sequence')
flags.DEFINE_integer('buffer_size', 128, 'Buffer size for dataset shuffling')
flags.DEFINE_boolean('use_xla', False, 'Whether or not to use XLA acceleration')
flags.DEFINE_boolean('use_amp', False, 'Whether or not to use Auto Mixed Precision')
flags.DEFINE_integer('epochs', 3, 'Number of epochs to train for')
flags.DEFINE_string('task', 'sst-2', 'Which task to re-train on')
flags.DEFINE_boolean('force_train', False, 'Set to True to overwrite an existing model')

def _set_config():
    """
    A helper function to set global options.
    """
    utils.set_up_environment()
    tf.config.optimizer.set_jit(FLAGS.use_xla)
    tf.config.optimizer.set_experimental_options({"auto_mixed_precision": FLAGS.use_amp})

def _get_tfds_task(task):
    """
    A helper function for getting the right
    task name.
    Args:
        task: The huggingface task name.
    """
    if task == "sst-2":
        return "sst2"
    elif task == "sts-b":
        return "stsb"
    return task

def train(argv=None):
    """
    A function that re-trains BERT for sentiment analysis.
    """
    _set_config()

    num_labels = len(glue_processors[FLAGS.task]().get_labels())

    # Load tokenizer and model from pretrained model/vocabulary.
    # Specify the number of labels to classify (2+: classification, 1: regression)
    config = BertConfig.from_pretrained("bert-base-cased",
                                        num_labels=num_labels,
                                        hidden_size=FLAGS.hidden_size)
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    model = TFBertForSequenceClassification.from_pretrained('bert-base-cased', config=config)

    # Load dataset via TensorFlow Datasets
    data, info = tensorflow_datasets.load(f'glue/{_get_tfds_task(FLAGS.task)}', with_info=True)
    train_examples = info.splits['train'].num_examples

    # MNLI expects either validation_matched or validation_mismatched
    valid_examples = info.splits['validation'].num_examples

    # Prepare dataset for GLUE as a tf.data.Dataset instance
    train_dataset = glue_convert_examples_to_features(data['train'],
                                                      tokenizer,
                                                      FLAGS.max_length,
                                                      FLAGS.task)

    # MNLI expects either validation_matched or validation_mismatched
    valid_dataset = glue_convert_examples_to_features(data['validation'],
                                                      tokenizer,
                                                      FLAGS.max_length,
                                                      FLAGS.task)
    train_dataset = train_dataset.shuffle(FLAGS.buffer_size).batch(FLAGS.batch_size).repeat(-1)
    valid_dataset = valid_dataset.batch(FLAGS.batch_size * 2)

    # Prepare training: Compile tf.keras model with optimizer, loss and learning rate schedule
    opt = tf.keras.optimizers.Adam(learning_rate=FLAGS.learning_rate, epsilon=FLAGS.epsilon)
    if FLAGS.use_amp:
        # loss scaling is currently required when using mixed precision
        opt = tf.keras.mixed_precision.experimental.LossScaleOptimizer(opt, 'dynamic')

    if num_labels == 1:
        loss = tf.keras.losses.MeanSquaredError()
    else:
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')

    model_path = f'./{_get_tfds_task(FLAGS.task)}/'

    if os.path.exists(model_path + 'tf_model.h5') and not FLAGS.force_train:
        print(f'Model in {model_path} already exists. Skipping training. ' + \
              'If you would like to force a re-train, set the force_train flag.')
        local_vars = locals()
        for variable in local_vars:
            if not variable.startswith('-'):
                print(f'{variable}:\t{local_vars[variable]}')
        return

    model.compile(optimizer=opt, loss=loss, metrics=[metric])

    # Train and evaluate using tf.keras.Model.fit()
    train_steps = train_examples // FLAGS.batch_size
    valid_steps = valid_examples // (FLAGS.batch_size * 2)

    _ = model.fit(train_dataset, epochs=FLAGS.epochs, steps_per_epoch=train_steps,
                  validation_data=valid_dataset, validation_steps=valid_steps)

    # Save TF2 model

#     os.makedirs(model_path, exist_ok=True)
    model.save_pretrained(model_path)

if __name__ == '__main__':
    app.run(train)

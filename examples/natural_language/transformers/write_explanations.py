import tensorflow as tf
import tensorflow_datasets
import numpy as np

from path_explain.path_explainer_tf import PathExplainerTF
from transformers import *
from tqdm import tqdm

from absl import app
from absl import flags
from path_explain import utils

FLAGS = flags.FLAGS
flags.DEFINE_string('task', 'sst-2', 'Which task to interpret')
flags.DEFINE_integer('batch_size', 32, 'Batch size for training')
flags.DEFINE_integer('num_samples', 128, 'Number of samples to draw when computing attributions')
flags.DEFINE_integer('max_length', 128, 'The maximum length of any sequence')
flags.DEFINE_boolean('get_attributions', False, 'Set to true to generate attributions')
flags.DEFINE_boolean('get_interactions', False, 'Set to true to generate interactions')

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

def interpret(argv=None):
    print('Loading model...')
    file = f'{_get_tfds_task(FLAGS.task)}/'
    num_labels = len(glue_processors[FLAGS.task]().get_labels())
    config = BertConfig.from_pretrained(file,
                                        num_labels=num_labels)
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    model = TFBertForSequenceClassification.from_pretrained(file,
                                                            config=config)

    print('Loading data...')
    data, info = tensorflow_datasets.load(f'glue/{_get_tfds_task(FLAGS.task)}',
                                          with_info=True)

    valid_dataset = glue_convert_examples_to_features(data['validation'],
                                                      tokenizer,
                                                      max_length=FLAGS.max_length,
                                                      task=FLAGS.task)
    valid_dataset = valid_dataset.batch(32)

    for batch in valid_dataset.take(1):
        batch_input = batch[0]
        batch_labels = batch[1]

    def embedding_model(batch_ids):
        batch_token_types = tf.zeros(batch_ids.shape)

        batch_embedding = model.bert.embeddings([batch_ids,
                                                 None,
                                                 batch_token_types])
        return batch_embedding

    batch_ids = batch_input['input_ids']
    batch_embedding = embedding_model(batch_ids)

    baseline_ids = np.zeros((1, 128), dtype=np.int64)
    baseline_embedding = embedding_model(baseline_ids)

    def prediction_model(batch_embedding):
        extended_attention_mask = tf.zeros(batch_embedding.shape[:2])
        extended_attention_mask = extended_attention_mask[:, tf.newaxis, tf.newaxis, :]
        extended_attention_mask = tf.cast(extended_attention_mask, tf.float32)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        head_mask = [None] * model.bert.num_hidden_layers

        batch_encoded = model.bert.encoder([batch_embedding,
                                                 extended_attention_mask,
                                                 head_mask])
        batch_sequence = batch_encoded[0]
        batch_pooled   = model.bert.pooler(batch_sequence)
        batch_predictions = model.classifier(batch_pooled)
        return batch_predictions

    output_index = 0
    if FLAGS.task == 'sst-2':
        output_index = 1
    explainer = PathExplainerTF(prediction_model)

    if FLAGS.get_attributions:
        print('Getting attributions...')
        attributions = explainer.attributions(inputs=batch_embedding,
                                              baseline=baseline_embedding,
                                              batch_size=FLAGS.batch_size,
                                              num_samples=FLAGS.num_samples,
                                              use_expectation=False,
                                              output_indices=output_index,
                                              verbose=True)
        np.save(f'{_get_tfds_task(FLAGS.task)}/attributions.npy', attributions)

    if FLAGS.get_interactions:
        print('Getting interactions...')

        for i in range(batch_ids.shape[0]):
            print('Sentence {}/{}'.format(i, batch_embedding.shape[0]))
            select_embedding = batch_embedding[i:i+1]
            select_ids = batch_ids[i]

            max_id_index = int(tf.where(select_ids == 0)[0])


            shape_tuple = tuple([int(a) for a in select_embedding.shape[1:]]) + \
                          (max_id_index, int(batch_embedding.shape[-1]))
            interaction_matrix = np.zeros(shape_tuple)

            for id_index in tqdm(range(max_id_index)):
                for embedding_index in range(batch_embedding.shape[-1]):
                    interactions = explainer.interactions(inputs=select_embedding,
                                                          baseline=baseline_embedding,
                                                          batch_size=FLAGS.batch_size,
                                                          num_samples=FLAGS.num_samples,
                                                          use_expectation=False,
                                                          output_indices=output_index,
                                                          verbose=False,
                                                          interaction_index=(id_index, embedding_index))
                    interaction_matrix[:, :, id_index, embedding_index] = interactions
            np.save(f'{_get_tfds_task(FLAGS.task)}/interactions_{i}.npy', interaction_matrix)

if __name__ == '__main__':
    app.run(interpret)
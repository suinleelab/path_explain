import tensorflow as tf
import tensorflow_datasets
import numpy as np

from embedding_explainer import EmbeddingExplainerTF
from transformers import *
from tqdm import tqdm

from absl import app
from absl import flags
from path_explain import utils

FLAGS = flags.FLAGS
flags.DEFINE_string('task', 'sst-2', 'Which task to interpret')
flags.DEFINE_integer('batch_size',  16, 'Batch size for interpretation')
flags.DEFINE_integer('num_samples', 256, 'Number of samples to draw when computing attributions')
flags.DEFINE_integer('max_length',  128, 'The maximum length of any sequence')
flags.DEFINE_boolean('get_attributions', False, 'Set to true to generate attributions')
flags.DEFINE_boolean('get_interactions', False, 'Set to true to generate interactions')
flags.DEFINE_boolean('custom_sentences', False, 'Set to true to explain a list of custom sentences')

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
    config = DistilBertConfig.from_pretrained(file,
                                              num_labels=num_labels)
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = TFDistilBertForSequenceClassification.from_pretrained(file,
                                                            config=config)


    if FLAGS.custom_sentences:
        print('Tokenizing custom sentences...')
        sentences = [
            'This movie was bad',
            'This movie was not bad',
            'A movie',
            'A bad movie',
            'A bad, terrible movie',
            'A bad, terrible, awful movie',
            'A bad, terrible, awful, horrible movie'
        ]

        input_ids_all = []
        token_type_ids_all = []
        attention_mask_all = []

        for sentence in sentences:
            encoded_sentence = tokenizer.encode_plus(sentence,
                                                     add_special_tokens=True,
                                                     return_tensors='tf',
                                                     pad_to_max_length=True,
                                                     max_length=FLAGS.max_length)
            input_ids = encoded_sentence['input_ids']
            token_type_ids = encoded_sentence['token_type_ids']
            attention_mask = encoded_sentence['attention_mask']

            input_ids_all.append(input_ids)
            token_type_ids_all.append(token_type_ids)
            attention_mask_all.append(attention_mask)

        batch_input = {
            'input_ids': tf.concat(input_ids_all, axis=0),
            'token_type_ids': tf.concat(token_type_ids_all, axis=0),
            'attention_mask': tf.concat(attention_mask_all, axis=0)
        }
    else:
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

    def embedding_model(batch_ids):
        batch_embedding = model.distilbert.embeddings(batch_ids)
        return batch_embedding

    batch_ids = batch_input['input_ids']
    batch_embedding = embedding_model(batch_ids)

    baseline_ids = np.zeros((1, 128), dtype=np.int64)
    baseline_embedding = embedding_model(baseline_ids)

    def prediction_model(batch_embedding):
        # Note: this isn't exactly the right way to use the attention mask.
        # It should actually indicate which words are real words. This
        # makes the coding easier however, and the output is fairly similar.
        attention_mask = tf.ones(batch_embedding.shape[:2])
        attention_mask = tf.cast(attention_mask, dtype=tf.float32)
        head_mask = [None] * model.distilbert.num_hidden_layers

        transformer_output = model.distilbert.transformer([batch_embedding, attention_mask, head_mask], training=False)[0]
        pooled_output = transformer_output[:, 0]
        pooled_output = model.pre_classifier(pooled_output)
        logits = model.classifier(pooled_output)
        return logits

    output_index = 0
    if FLAGS.task == 'sst-2':
        output_index = 1
    explainer = EmbeddingExplainerTF(prediction_model)

    if FLAGS.get_attributions:
        print('Getting attributions...')
        attributions = explainer.attributions(inputs=batch_embedding,
                                              baseline=baseline_embedding,
                                              batch_size=FLAGS.batch_size,
                                              num_samples=FLAGS.num_samples,
                                              use_expectation=False,
                                              output_indices=output_index,
                                              verbose=True)

        if FLAGS.custom_sentences:
            np.save(f'{_get_tfds_task(FLAGS.task)}/attributions_custom.npy', attributions)
        else:
            np.save(f'{_get_tfds_task(FLAGS.task)}/attributions.npy', attributions)

    if FLAGS.get_interactions:
        print('Getting interactions...')
        interactions = explainer.interactions(inputs=batch_embedding,
                                              baseline=baseline_embedding,
                                              batch_size=FLAGS.batch_size,
                                              num_samples=FLAGS.num_samples,
                                              use_expectation=False,
                                              output_indices=output_index,
                                              verbose=True)
        if FLAGS.custom_sentences:
            np.save(f'{_get_tfds_task(FLAGS.task)}/interactions_custom.npy', interactions)
        else:
            np.save(f'{_get_tfds_task(FLAGS.task)}/interactions.npy', interactions)


if __name__ == '__main__':
    app.run(interpret)
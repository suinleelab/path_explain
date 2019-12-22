import tensorflow as tf
import tensorflow_datasets
import numpy as np

from bert_explainer import BertExplainerTF
from transformers import *
from tqdm import tqdm

from absl import app
from absl import flags
from path_explain import utils

FLAGS = flags.FLAGS
flags.DEFINE_string('task', 'sst-2', 'Which task to re-train on')
flags.DEFINE_integer('batch_size', 32, 'Batch size for training')
flags.DEFINE_integer('num_samples', 512, 'Number of samples to draw when computing attributions')
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

def _get_interactions_for_ids(batch_ids,
                              batch_baseline,
                              tokenizer,
                              explainer):
    """
    Helper function to get interactions for only the relevant tokens.

    Args:
        batch_ids: An array of encoded ids to be fed into a model.
        batch_baseline: The baseline to be fed into the model.
        tokenizer: A transformers tokenizer object.
        explainer: A bert explainer object.
    """
    batch_tokens = tokenizer.convert_ids_to_tokens(batch_ids.numpy())
    start_index = 1
    end_index = batch_tokens.index('[SEP]')
    interaction_array = []
    for index_to_explain in tqdm(range(start_index, end_index)):
        interactions = explainer.interactions(inputs=np.expand_dims(batch_ids, axis=0),
                                              baseline=batch_baseline,
                                              batch_size=FLAGS.batch_size,
                                              num_samples=FLAGS.num_samples,
                                              use_expectation=False,
                                              output_indices=1,
                                              verbose=False,
                                              interaction_index=int(index_to_explain))
        interaction_array.append(interactions)
    interactions = np.concatenate(interaction_array, axis=0)
    return interactions

def _join_tokens(tokens, attributions, perform_copy=False):
    """
    Internal function to join tokens encoded
    in the huggingface tokenization style.

    Args:
        tokens: An array of strings.
        attributions: A numpy array of importances.
        perform_copy: Set to True to join interactions.
    """
    joined_attributions = []
    joined_tokens = []
    current_string = ''
    current_attribution = 0
    found_special = False
    add_last = False

    for i, token in enumerate(tokens):
        if token == '[CLS]' or \
           token == '[SEP]' or \
           token == '[PAD]':
            continue

        if perform_copy:
            attr = attributions[i].copy()
        else:
            attr = attributions[i]

        add_last = False
        if token.startswith('##'):
            current_string += token[2:]
            current_attribution += attr
        elif token in "'-":
            current_string += token
            current_attribution += attr
            found_special = True
        elif current_string == '' or found_special:
            current_string += token
            current_attribution += attr
            found_special = False
        else:
            joined_tokens.append(current_string)
            joined_attributions.append(current_attribution)
            current_string = token
            current_attribution = attr

            add_last = True
    if add_last:
        joined_tokens.append(current_string)
        joined_attributions.append(current_attribution)

    return joined_tokens, joined_attributions

def _join_interactions(batch_tokens, interactions):
    """
    A helper function to join interaction values across
    tokens that split a word into multiple pieces.

    Args:
        batch_tokens: An array of tokens
    """
    _, joined_interactions = _join_tokens(batch_tokens,
                                          interactions,
                                          perform_copy=True)
    joined_interactions = np.stack(joined_interactions, axis=0)
    joined_interactions = np.swapaxes(joined_interactions, 0, 1)
    _, joined_interactions = _join_tokens(batch_tokens,
                                          joined_interactions,
                                          perform_copy=True)
    joined_interactions = np.stack(joined_interactions, axis=0)
    joined_interactions = np.swapaxes(joined_interactions, 0, 1)
    return joined_interactions

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
    valid_dataset = valid_dataset.batch(16)

    for batch in valid_dataset.take(1):
        batch_input = batch[0]
        batch_labels = batch[1]

    batch_ids = batch_input['input_ids']
    batch_baseline = np.zeros((1, 128))

    explainer = BertExplainerTF(model)

    if FLAGS.get_attributions:
        print('Getting attributions...')
        attributions = explainer.attributions(inputs=batch_ids,
                                              baseline=batch_baseline,
                                              batch_size=FLAGS.batch_size,
                                              num_samples=FLAGS.num_samples,
                                              use_expectation=False,
                                              output_indices=1,
                                              verbose=True)
        for i in range(batch_ids.shape[0]):
            attribution = attributions[i]
            batch_tokens = tokenizer.convert_ids_to_tokens(batch_ids[i].numpy())
            joined_tokens, joined_attributions = _join_tokens(batch_tokens, attribution)
            joined_tokens = np.array(joined_tokens)
            np.save('{}/attribution_{}.npy'.format(_get_tfds_task(FLAGS.task),
                                                   i), joined_attributions)
            np.save('{}/tokens_{}.npy'.format(_get_tfds_task(FLAGS.task),
                                              i), joined_tokens)

    if FLAGS.get_interactions:
        print('Getting interactions...')
        for i in range(batch_ids.shape[0]):
            print('Sentence {}/{}'.format(i, batch_ids.shape[0]))
            batch_id = batch_ids[i]
            batch_tokens = tokenizer.convert_ids_to_tokens(batch_id.numpy())
            start_index = 1
            end_index = batch_tokens.index('[SEP]')
            batch_interactions = _get_interactions_for_ids(batch_id,
                                                           batch_baseline,
                                                           tokenizer,
                                                           explainer)
            joined_interactions = _join_interactions(batch_tokens[start_index:end_index],
                                                     batch_interactions)
            np.save('{}/interactions_{}.npy'.format(_get_tfds_task(FLAGS.task), i), batch_interactions)
            np.save('{}/joined_interactions_{}.npy'.format(_get_tfds_task(FLAGS.task), i), joined_interactions)

if __name__ == '__main__':
    app.run(interpret)
"""
A module for setting up tensorflow gpu
environments.
"""
import os
import tensorflow as tf
import numpy as np

def set_up_environment(mem_frac=None, visible_devices=None, min_log_level='3'):
    """
    A helper function to set up a tensorflow environment.

    Args:
        mem_frac: Fraction of memory to limit the gpu to. If set to None,
                  turns on memory growth instead.
        visible_devices: A string containing a comma-separated list of
                         integers designating the gpus to run on.
        min_log_level: One of 0, 1, 2, or 3.
    """
    if visible_devices is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = visible_devices

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(min_log_level)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                if mem_frac is not None:
                    memory_limit = int(10000 * mem_frac)
                    config = [tf.config.experimental.VirtualDeviceConfiguration(
                        memory_limit=memory_limit)]
                    tf.config.experimental.set_virtual_device_configuration(gpu, config)
                else:
                    tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as error:
            print(error)

def softplus_activation(beta=1.0):
    """
    Returns a callable function
    representing softplus activation
    with beta smoothing.

    Args:
        beta: The smoothing parameter. Smaller means smoother.
    """
    def softplus(batch_x):
        return (1.0 / beta) * tf.keras.backend.log(1.0 + \
               tf.keras.backend.exp(-1.0 * tf.keras.backend.abs(beta * batch_x))) + \
               tf.keras.backend.maximum(batch_x, 0)
    return softplus

def _find_sublist(super_list, sub_list):
    """
    A helper funcion to find the location of sub_lists in a super_list.
    """
    indices = []
    for i in range(len(super_list)):
        if super_list[i:i+len(sub_list)] == sub_list:
            indices.append([i + j for j in range(len(sub_list))])
    return indices

def _find_step_increasing(index_list):
    """
    A helper function to find the sequential sublists
    of a list.
    """
    indices = []
    current_start = index_list[0]

    for i in range(1, len(index_list)):
        if index_list[i] != index_list[i - 1] + 1:
            indices.append((current_start - 1, index_list[i - 1] + 1))
            current_start = index_list[i]

    indices.append((current_start - 1, index_list[-1] + 1))
    return indices

def fold_array(array, join_ranges):
    """
    A helper function to fold a numpy array along certain ranges.
    """
    array = array.copy()
    delete_slices = []
    for join_range in join_ranges:
        array[join_range[0]] = np.sum(array[join_range[0]:join_range[1]])
        delete_slices.append(np.arange(join_range[0] + 1, join_range[1]))
    delete_slices = np.concatenate(delete_slices, axis=0)

    array = np.delete(array, delete_slices)
    return array

def fold_matrix(array, join_ranges):
    """
    A helper function to fold a number matrix along certain ranges.
    """
    array = array.copy()
    delete_slices = []
    for join_range in join_ranges:
        array[join_range[0], :] = np.sum(array[join_range[0]:join_range[1], :], axis=0)
        array[:, join_range[0]] = np.sum(array[:, join_range[0]:join_range[1]], axis=1)

        delete_slices.append(np.arange(join_range[0] + 1, join_range[1]))
    delete_slices = np.concatenate(delete_slices, axis=0)

    array = np.delete(array, delete_slices, axis=0)
    array = np.delete(array, delete_slices, axis=1)
    return array

def fold_tokens(array, join_ranges, join_string='##'):
    """
    A helper function to fold a list of strings along certain ranges.
    """
    delete_slices = []
    for join_range in join_ranges:
        replace_string = ''.join(list(array[join_range[0]:join_range[1]])).replace(join_string, '')
        array[join_range[0]] = replace_string
        delete_slices.append(np.arange(join_range[0] + 1, join_range[1]))
    delete_slices = np.concatenate(delete_slices, axis=0)

    array = np.delete(array, delete_slices)
    return array

def strip_tokens(tokens,
                 attributions,
                 interactions,
                 start_character='[CLS]',
                 end_character='[SEP]',
                 join_string='##',
                 special_strings=[['n', "'", 't'], ["'", 's']]):
    """
    A helper function to strip and re-arrange attributions generated
    through huggingface transformer models.

    Args:
        tokens: a numpy matrix of words, shaped [batch_size, sequence_length]
        attributions: a numpy matrix of floats, shaped [batch_size, sequence_length]
        interactions: a numpy matrix of floats, shaped [batch_size,
                      sequence_length, sequence_length]
    """
    token_list = []
    attribution_list = []
    interaction_list = []

    for i in range(attributions.shape[0]):
        token = tokens[i]
        attribution = attributions[i]
        interaction = interactions[i]

        start_index = token.index(start_character) + 1
        end_index = token.index(end_character)

        token = token[start_index:end_index]
        attribution = attribution[start_index:end_index]
        interaction = interaction[start_index:end_index, start_index:end_index]

        if join_string is not None:
            join_indices = np.where([t.startswith(join_string) for t in token])[0]
#             join_indices = np.append(join_indices, join_indices - 1)

            if special_strings is not None:
                for special_string in special_strings:
                    special_string_indices = _find_sublist(token, special_string)
                    join_indices = np.append(join_indices, special_string_indices)

            if len(join_indices) > 0:
                join_indices = np.sort(np.unique(join_indices)).astype(int)
                join_ranges = _find_step_increasing(join_indices)

                attribution = fold_array(attribution, join_ranges)
                interaction = fold_matrix(interaction, join_ranges)
                token = fold_tokens(token, join_ranges)

        token_list.append(token)
        attribution_list.append(attribution)
        interaction_list.append(interaction)

    return token_list, attribution_list, interaction_list

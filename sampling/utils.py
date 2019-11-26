import numpy as np
import tensorflow as tf
import os

def normalize(im_batch, _range=None, _domain=None):
    if len(im_batch.shape) == 2:
        axis = (0, 1)
    elif len(im_batch.shape) == 3:
        axis = (0, 1, 2)
    elif len(im_batch.shape) == 4:
        axis = (1, 2, 3)
    else:
        raise ValueError('im_batch must be of rank 2, 3 or 4')
    
    if _domain is not None:
        min_vals = _domain[0]
        max_vals = _domain[1]
    else:
        min_vals = np.amin(im_batch, axis=axis, keepdims=True)
        max_vals = np.amax(im_batch, axis=axis, keepdims=True)
    
    norm_batch = (im_batch - min_vals) / (max_vals - min_vals)
    
    if _range is not None:
        amin = _range[0]
        amax = _range[1]
        norm_batch = norm_batch * (amax - amin) + amin
    return norm_batch

def batch_standardize(x, multi_class=True):
    if multi_class:
        mean_per_row = np.mean(x, axis=(1, 2, 3, 4), keepdims=True)
        std_per_row  = np.std(x,  axis=(1, 2, 3, 4), keepdims=True)
        return (x - mean_per_row) / std_per_row
    else:
        mean_per_row = np.mean(x, axis=(1, 2, 3), keepdims=True)
        std_per_row  = np.std(x,  axis=(1, 2, 3), keepdims=True)
        return (x - mean_per_row) / std_per_row

def set_up_environment(mem_frac=None):
#     tf.enable_eager_execution()
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                if mem_frac is not None:
                    memory_limit = int(10000 * mem_frac)
                    tf.config.experimental.set_virtual_device_configuration(
                        gpu,
                    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit)])
                else:
                    tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
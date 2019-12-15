"""
A module for setting up tensorflow gpu
environments.
"""
import os
import tensorflow as tf

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

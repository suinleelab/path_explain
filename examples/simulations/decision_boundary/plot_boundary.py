import tensorflow as tf
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn import datasets, model_selection

from path_explain.utils import set_up_environment
from absl import app
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_integer('batch_size', 100, 'Batch size for training')
flags.DEFINE_integer('epochs', 30, 'Number of epochs to train for')
flags.DEFINE_integer('dataset_size', 1000, 'Number of points to generate')


flags.DEFINE_float('learning_rate', 0.3, 'Learning rate to use while training')
flags.DEFINE_float('factor', 0.5, 'Separation between classes')
flags.DEFINE_float('noise', 0.15, 'Amount of within-class noise')


flags.DEFINE_string('visible_devices', '0', 'Which gpu to train on')

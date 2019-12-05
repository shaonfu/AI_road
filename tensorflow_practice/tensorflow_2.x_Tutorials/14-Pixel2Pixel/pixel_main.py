import os
import tensorflow as tf
import numpy as np
from  tensorflow import keras
import time
from  matplotlib import pyplot as plt

from  gd import Discriminator, Generator

tf.random.set_seed(22)
np.random.seed(22)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
assert tf.__version__.startswith('2.')
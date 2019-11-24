import os
import time
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import summary_ops_v2
from tensorflow import keras
from tensorflow.keras import datasets,layers,models,optimizers,metrics

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def mnist_datasets():
    (x_train,y_train),(x_test,y_test) = datasets.mnist.load_data()
    # Numpy defaults to dtype=float64; TF defaults to float32. Stick with float32.
    x_train,x_test = x_train / np.float32(255),x_test / np.float32(255)
    y_train,y_test = y_train.astype(np.int64) , y_test.astype(np.int64)
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train,y_train))
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test,y_test))
    return train_dataset, test_dataset

train_ds = mnist_datasets()
train_ds = train_ds.shuffle(60000).batch(100)
test_ds = test_ds.batch(100)

model = tf.keras.Sequential([
    layers.Reshape(
        target_shape = [28,28,1],
        input_shape = (28,28,)
    ),
    layers.Conv2D(2,5,padding ='same',activation = tf.nn.relu),
    layers.MaxPooling2D((2,2),(2,2),padding='same'),
    layers.Conv2D(4,5,padding ='same',activation = tf.nn.relu),
    layers.MaxPooling2D((2,2),(2,2),padding='same'),
    layers.Flatten(),
    layers.Dense(32,activation = tf.nn.relu),
    layers.Dropout(rate = 0.25)
    layers.Dense(10,activation = tf.nn.softmax)
])

optimizer = optimizers.SGD(learning_rate = 0.01,momentum = 0.5)

compute_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)
compute_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
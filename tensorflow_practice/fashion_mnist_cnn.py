#%%
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers
tfds.disable_progress_bar()

import math
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

#%%
import logging
logger = logging.getLogger()
logger.setLevel(logging.ERROR)


#%%
dataset,metadata = tfds.load('fashion_mnist',as_supervised=True,with_info=True)
train_dataset,test_dataset = dataset['train'],dataset['test']

#%%
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal',      'Shirt',   'Sneaker',  'Bag',   'Ankle boot']

#%%
num_train_examples = metadata.splits['train'].num_examples
num_test_examples = metadata.splits['test'].num_examples
print("Number of training examples: {}".format(num_train_examples))
print("Number of test examples:     {}".format(num_test_examples))

#%%
def normalize(images, labels):
  images = tf.cast(images, tf.float32)
  images /= 255
  return images, labels

# The map function applies the normalize function to each element in the train
# and test datasets
train_dataset =  train_dataset.map(normalize)
test_dataset  =  test_dataset.map(normalize)

#%%
for image ,label in test_dataset.take(1):
    break
image = image.numpy().reshape((28,28))

plt.figure()
plt.imshow(image,cmap=plt.cm.binary)
plt.colorbar()
plt.grid(False)
plt.show()

#%%
plt.figure(figsize=(10,10))
i = 0
for (image,label) in test_dataset.take(25):
    image = image.numpy().reshape((28,28))
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(image, cmap=plt.cm.binary)
    plt.xlabel(class_names[label])
    i+=1
plt.show()

#%%
model = tf.keras.Sequential([
    layers.Conv2D(32,(3,3),padding='same',activation=tf.nn.relu,input_shape=(28,28,1)),
    layers.MaxPooling2D((2,2),strides=2),
    layers.Conv2D(64,(3, 3), padding = "same", activation = tf.nn.relu),
    layers.MaxPooling2D((2,2),strides=2),
    layers.Flatten(),
    layers.Dense(128,activation=tf.nn.relu),
    layers.Dense(10,activation=tf.nn.softmax)
])

model.compile(optimizer='adam',
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])

#%%
model.summary()

#%%
BATCH_SIZE = 32
train_dataset = train_dataset.repeat().shuffle(num_train_examples).batch(BATCH_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)

model.fit(train_dataset,epochs=5,steps_per_epoch=math.ceil(num_train_examples/BATCH_SIZE))

#%%
test_loss,test_acc = model.evaluate(test_dataset,steps=math.ceil(num_test_examples/BATCH_SIZE))
print(test_loss,test_acc)

#%%

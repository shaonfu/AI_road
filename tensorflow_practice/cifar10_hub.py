#%%
from __future__ import absolute_import, division, print_function, unicode_literals

import matplotlib.pylab as plt
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub

from tensorflow.keras import layers

#%%
mobilenet_URL = "models/mobileNet"

IMAGE_SHAPE = (224,224)
feature_extractor = tf.keras.Sequential([
    hub.KerasLayer(mobilenet_URL, input_shape=IMAGE_SHAPE+(3,))
])


#%%
splits = tfds.Split.ALL.subsplit(weighted=(80,20))

splits,info = tfds.load('cifar10',with_info=True,download=False,data_dir='.\datas',as_supervised=True,split = splits)

#%%
(train_examples,validation_examples) = splits 

num_examples = info.splits['train'].num_examples
nun_classes = info.features['label'].num_classes


#%%
for i, example_image in enumerate(train_examples.take(3)):
  print("Image {} shape: {}".format(i+1, example_image[0].shape))

#%%
def format_image(image,label):
  image = tf.image.resize(image,(224,224)) / 255.0
  return image,label

BATCH_SIZE = 32

train_batches = train_examples.shuffle(num_examples//4).map(format_image).batch(BATCH_SIZE).prefetch(1)
validation_batches = validation_examples.map(format_image).batch(BATCH_SIZE).prefetch(1)

#%%
image_batch, label_batch = next(iter(train_batches.take(1)))
image_batch = image_batch.numpy()
label_batch = label_batch.numpy()

#%%
feature_batch = feature_extractor(image_batch)
print(feature_batch.shape)

feature_extractor.trainable = False

#%%
model = tf.keras.Sequential([
  feature_extractor,
  layers.Dense(10, activation='softmax')
])

model.summary()

#%%
model.compile(
  optimizer='adam',
  loss='sparse_categorical_crossentropy',
  metrics=['accuracy'])

EPOCHS = 30
history = model.fit(train_batches,
                    epochs=EPOCHS,
                    validation_data=validation_batches)


# %%

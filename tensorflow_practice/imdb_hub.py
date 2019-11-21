#%%
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np

import tensorflow as tf

import tensorflow_hub as hub
import tensorflow_datasets as tfds

print("Version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())
print("Hub version: ", hub.__version__)
print("GPU is", "available" if tf.config.experimental.list_physical_devices("GPU") else "NOT AVAILABLE")


#%%
# 将训练集按照 6:4 的比例进行切割，从而最终我们将得到 15,000
# 个训练样本, 10,000 个验证样本以及 25,000 个测试样本
train_validation_split = tfds.Split.TRAIN.subsplit([6, 4])

(train_data,validation_data),test_data = tfds.load(
    name="imdb_reviews",
    data_dir = "./data",
    split = (train_validation_split,tfds.Split.TEST),
    as_supervised=True
)

#%%
train_examples_batch,train_labels_batch = next(iter(train_data.batch(10)))
train_examples_batch

#%%
embedding = "models/gnews-swivel-20dim"
hub_layer = hub.KerasLayer(embedding, input_shape=[], 
                           dtype=tf.string, trainable=True)
hub_layer(train_examples_batch[:3])

#%%
model = tf.keras.Sequential([
    hub_layer,
    tf.keras.layers.Dense(16,activation='relu'),  
    tf.keras.layers.Dense(1,activation='sigmoid')    
])

#%%
model.summary()

#%%
model.compile(optimizer = 'adam',
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])

#%%
history = model.fit(train_data.shuffle(10000).batch(512),
                    epochs = 20,validation_data = validation_data.batch(512),
                    verbose = 1)

#%%
results = model.evaluate(test_data.batch(512),verbose = 2)
for name,value in zip(model.metrics_names,results):
    print("%s: %.3f" % (name, value))

#%%
history_dict = history.history
history_dict.keys()

#%%
import matplotlib.pyplot as plt

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

# "bo" is for "blue dot"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

#%%
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

#%%

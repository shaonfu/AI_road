#%%
from __future__ import absolute_import,division,print_function,unicode_literals

import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

#%%
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '/gpu:0'

#%%
import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

#%%
_URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
zip_dir = tf.keras.utils.get_file('cats_and_dogs_filterted.zip',origin=_URL,cache_dir='./data',extract=True)

#%%
train_dir = './datas/datasets/cats_and_dogs_filtered/train'
train_cats_dir = 'datas/datasets/cats_and_dogs_filtered/train/cats'
train_dogs_dir = 'datas/datasets/cats_and_dogs_filtered/train/dogs'
validation_dir = 'datas/datasets/cats_and_dogs_filtered/validation'
validation_cats_dir = './datas/datasets/cats_and_dogs_filtered/validation/cats'
validation_dogs_dir = './datas/datasets/cats_and_dogs_filtered/validation/dogs'

num_cats_tr = len(os.listdir(train_cats_dir))
num_dogs_tr = len(os.listdir(train_dogs_dir))

num_cats_val = len(os.listdir(validation_cats_dir))
num_dogs_val = len(os.listdir(validation_dogs_dir))

#%%
total_train = num_cats_tr + num_dogs_tr
total_val = num_cats_val + num_dogs_val

#%%
print('total training cat images:', num_cats_tr)
print('total training dog images:', num_dogs_tr)

print('total validation cat images:', num_cats_val)
print('total validation dog images:', num_dogs_val)
print("--")
print("Total training images:", total_train)
print("Total validation images:", total_val)

#%%
BATCH_SIZE = 100
IMG_SHAPE = 150

#%%
train_image_generator = ImageDataGenerator(rescale=1./255) 
validation_image_generator = ImageDataGenerator(rescale=1./255) 

#%%
train_data_gen = train_image_generator.flow_from_directory(
    batch_size = BATCH_SIZE,
    directory = train_dir,
    shuffle = False,
    target_size = (IMG_SHAPE,IMG_SHAPE),
    class_mode='binary')

#%%
validation_data_gen = train_image_generator.flow_from_directory(
    batch_size = BATCH_SIZE,
    directory = validation_dir,
    shuffle = False,
    target_size = (IMG_SHAPE,IMG_SHAPE),
    class_mode='binary')

#%%
sample_training_images, _ = next(train_data_gen)

#%%
def plotImages(images_arr):
    fig,axes = plt.subplots(1,5,figsize=(20,20))
    axes = axes.flatten()
    for img,ax in zip(images_arr,axes):
        ax.imshow(img)
    plt.tight_layout()
    plt.show()

plotImages(sample_training_images[:5])

#%%
#Flipping the image horizontally
image_gen = ImageDataGenerator(rescale = 1./255,horizontal_flip = True)

train_data_gen = image_gen.flow_from_directory(batch_size = BATCH_SIZE,
                                               directory = train_dir,
                                               shuffle = True,
                                               target_size = (IMG_SHAPE, IMG_SHAPE))

#%%
augmented_images = [train_data_gen[0][0][0] for i in range(5)]
plotImages(augmented_images)

#%%
#Rotating the image
image_gen = ImageDataGenerator(rescale = 1./255,rotation_range=45)

train_data_gen = image_gen.flow_from_directory(batch_size = BATCH_SIZE,
                                               directory = train_dir,
                                               shuffle = True,
                                               target_size = (IMG_SHAPE, IMG_SHAPE))

#%%
augmented_images = [train_data_gen[0][0][0] for i in range(5)]
plotImages(augmented_images)

#%%
#Applying Zoom
image_gen = ImageDataGenerator(rescale = 1./255,zoom_range=0.5)

train_data_gen = image_gen.flow_from_directory(batch_size = BATCH_SIZE,
                                               directory = train_dir,
                                               shuffle = True,
                                               target_size = (IMG_SHAPE, IMG_SHAPE))

#%%
augmented_images = [train_data_gen[0][0][0] for i in range(5)]
plotImages(augmented_images)

#%%
#Putting it all together
image_gen_train = ImageDataGenerator(
      rescale=1./255,
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')

train_data_gen = image_gen_train.flow_from_directory(batch_size=BATCH_SIZE,
                                                     directory=train_dir,
                                                     shuffle=True,
                                                     target_size=(IMG_SHAPE,IMG_SHAPE),
                                                     class_mode='binary')

#%%
augmented_images = [train_data_gen[0][0][0] for i in range(5)]
plotImages(augmented_images)

#%%
image_gen_val = ImageDataGenerator(rescale=1./255)

val_data_gen = image_gen_val.flow_from_directory(batch_size=BATCH_SIZE,
                                                 directory=validation_dir,
                                                 target_size=(IMG_SHAPE, IMG_SHAPE),
                                                 class_mode='binary')

#%%
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')
])

#%%
model.compile(optimizer = 'adam',
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])

#%%
model.summary()

#%%
'''
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)
'''

#%%
"""
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
tf.compat.v1.enable_eager_execution(config=config)
"""
#%%
"""
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
tf.keras.backend.set_session(tf.compat.v1.Session(config = config))
"""

#%%
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
#config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.9
session = InteractiveSession(config=config)
tf.compat.v1.enable_eager_execution(config=config)

#%%
epochs = 5
history = model.fit_generator(
    train_data_gen,
    steps_per_epoch=int(np.ceil(total_train / float(BATCH_SIZE))),
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps=int(np.ceil(total_val / float(BATCH_SIZE)))
)

#%%
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

#%%
from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())

# %%
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# %%
value = tf.test.is_gpu_available(
    cuda_only=False,
    min_cuda_compute_capability=None
)
print ('***If TF can access GPU: ***\n\n',value) # MUST RETURN True IF IT CAN!!

# %%
tf.test.is_gpu_available()

# %%

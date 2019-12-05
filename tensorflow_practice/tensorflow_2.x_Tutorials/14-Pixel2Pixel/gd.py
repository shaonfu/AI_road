import os
import tensorflow as tf
import numpy as np
from tensorflow import keras

class Downsample(keras.Model):

    def __init__(self,filters,size,apply_batchnorm=True):
        super(Downsample,self).__init__

        self.apply_batchnorm = apply_batchnorm
        initializer = tf.random_normal_initializer(0.,0.02)

        self.conv1 = keras.layers.Conv2D(filters,
                                        (size,size),
                                         strides = 2,
                                         padding ='same',
                                         kernel_initializer = initializer,
                                         use_bias = False)
        if self.apply_batchnorm:
            self.apply_batchnorm = keras.layers.BatchNormalization()

    def call(self,x,training):
        x = self.conv1(x)
        if self.apply_batchnorm:
            x = self.apply_batchnorm(x,training=training)
        x = tf.nn.leaky_relu(x)
        return x

class Upsample(keras.Model):

    def __init__(self,filters,size,apple_dropout=False):
        super(Upsample,self).__init__()

        self.apply_dropout = apple_dropout
        initializer = tf.random_normal_initializer(0.,0.02) 

    def call(self,x1,x2,training=None):
        x = self.up_conv(x1)


        return x


class Generator(keras.Model):

    def __init__(self):
        super(Generator,self).__init__()

        
    def call(self,x,training = None):
        
        return x16


class DiscDownsample(keras.Model):

    def __init__(self,filters,size,apply_batchnorm=True):
        super(DiscDownsample,self).__init__()

    def call(self,x,training = None):
        x = self.conv1(x)

        return x

    
class Discriminator(keras.Model):

    def __init__(self):
        super(Discriminator,self).__init__() 

        initializer = tf.random_normal_initializer(0.,0.02)

    def call(self,inputs, training = None):
        inp,target = inputs

        #concatenating the input and the target

        return x
        
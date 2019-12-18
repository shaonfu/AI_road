import os
import tensorflow as tf
import numpy as np
from tensorflow import keras


tf.random.set_seed(22)
np.random.seed(22)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'
assert tf.__version__.startswith('2.')

(x_train,y_train),(x_test,y_test) = keras.datasets.fashion_mnist.load_data()
x_train,x_test = x_train.astype(np.float32)/255., x_test.astype(np.float32)/255.
#[b,28,28]=>[b,28,28,1]
x_train,x_test = np.expand_dims(x_train,axis=3),np.expand_dims(x_test,axis=3)

#one hot encode the labels. convert back to numpy as we cannot use a combination of numpy
#and tensors as input to keras
y_train_ohe = tf.one_hot(y_train,depth=10).numpy()
y_test_ohe = tf.one_hot(y_test,depth=10).numpy()

print(x_train.shape,y_train.shape, x_test.shape, y_test.shape)

#3x3 convolution
def conv3x3(channels,strides=1,kernel=(3,3)):
    return keras.layers.Conv2D(channels,kernel,strides=strides,padding='same',
                               use_bias = False,
                               kernel_initializer = tf.random_normal_initializer())



class ResnetBlock(keras.Model):

    def __init__(self,channels, strides = 1, residual_path = False):
        super(ResnetBlock,self).__init__()

        self.channels = channels
        self.strides = strides
        self.residual_path = residual_path

        self.conv1 = conv3x3(channels,strides)
        self.bn1 = keras.layers.BatchNormalization()
        self.conv2 = conv3x3(channels)
        self.bn2 = keras.layers.BatchNormalization()

        if residual_path:
            self.down_conv = conv3x3(channels,strides,kernel=(1, 1))
            self.down_bn = tf.keras.layers.BatchNormalization()

    def call(self,inputs,training = None):
        residual = inputs
        
        x = self.bn1(inputs,training=training)
        x = tf.nn.relu(x)
        x = self.conv1(x)
        x = self.bn2(x,training=training)
        x = tf.nn.relu(x)
        x = self.conv2(x)

        # this module can be added into self.
        # however, module in for can not be added.
        if self.residual_path:
            residual = self.down_bn(inputs,training=training)
            residual = tf.nn.relu(residual)
            residual = self.down_conv(residual)
        
        x = x + residual
        return x

class ResNet(keras.Model):

    def __init__(self,block_list,num_classes,initinal_filters=16,**kwargs):
        super(ResNet,self).__init__(**kwargs)

        self.num_blocks = len(block_list)
        self.block_list = block_list

        self.in_channels = initinal_filters
        self.out_channels = initinal_filters
        self.conv_initial = conv3x3(self.out_channels)

        self.blocks = keras.models.Sequential(name='dynamic-blocks')

        #build all the blocks
        for


    def call(self,inputs,training = None):
        
        return out

def main():
    num_classes = 10
    batch_size = 32
    epochs = 1

if __name__ == 'main':
    main()
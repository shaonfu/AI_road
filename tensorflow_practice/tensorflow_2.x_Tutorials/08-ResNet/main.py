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

    def __init__(self,channels, strides = 1, residual = False):
        super(ResnetBlock,self).__init__()

    def call(self,inputs,training = None):
        residual = inputs

        return x

class ResNet(keras.Model):

    def __init__(self,block_list,num_classes,initinal_filters=60,**kwargs):
        super(ResnetBlock,self).__init__(**kwargs)

    def call(self,inputs,training = None):
        
        return out

def main():
    num_classes = 10

if __name__ == 'main':
    main()
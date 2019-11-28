import os
import tensorflow as tf
import numpy as np
from tensorflow import keras

tf.random.set_seed(22)
np.random.seed(22)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
assert tf.__version__.startswith('2.')


#data预处理
(x_train,y_train) , (x_test,y_test) = keras.datasets.mnist.load_data()
x_train,x_test = x_train.astype(np.float32)/255., x_test.astype(np.float32)/255.
#[b,28,28]=>[b,28,28,1]
x_train,x_test = np.expand_dims(x_train,axis=3),np.expand_dims(x_test,axis=3)

db_train = tf.data.Dataset.from_tensor_slices((x_train,y_train)).batch(256)
db_test = tf.data.Dataset.from_tensor_slices((x_test,y_test)).batch(256)


print(x_train.shape,y_train.shape)
print(x_test.shape,y_test.shape)


class ConvBNRelu(keras.Model):
    def __str__(self,ch,kernel_size = 3,strides = 1, padding ='same'):
        super(ConvBNRelu,self).__init__()

        self.model = keras.models.Sequential([

        ])

    def call(self,x,training=None):

        x = self.model(x,training = training)

        return x
    

class InceptionBlk(keras.Model):

    def __init__(self,ch,strides=1):
        super(InceptionBlk,self).__init__()

        self.ch = ch
    
    def call(self,x,training=None):

        return x

class Inception(keras.Model):
    def __init__(self,num_layers,num_classes,init_ch=16,**kwargs):
        super(Inception,self).__init__(**kwargs)

    def call(self,x,training = None):
        out = self.conv1(x,training = training)



        return out
    
#build model and optimizer
batch_size = 32
epochs = 100


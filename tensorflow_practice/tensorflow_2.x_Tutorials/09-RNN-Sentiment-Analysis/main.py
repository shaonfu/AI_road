import os
import tensorflow as tf
import numpy as np
from tensorflow import keras


tf.random.set_seed(22)
np.random.seed(22)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'
assert tf.__version__.startswith('2.')

# fix random seed for reproducibility
np.random.seed(7)
# load the dataset but only keep the top n words,zero the rest
top_words = 10000
# truncate and pad input sequences
max_review_length = 80
(X_train,y_train),(X_test,y_test) = keras.datasets.imdb.load_data(num_words=top_words)
print('Pad sequences (samples x time)')
x_train = keras.preprocessing.sequence.pad_sequences(X_train,maxlen=max_review_length)
x_test = keras.preprocessing.sequence.pad_sequences(X_test,maxlen=max_review_length)
print(x_train.shape,x_test.shape)

class RNN(keras.Model):
    def __init__(self,units,num_class,num_classes):
        super(RNN,self).__init__()



    
    def call(self,inputs,training=None,mask=None):

        return x
    

def main():

    units = 64
    num_classes = 2
    batch_size = 32
    epochs = 20


if __name__ =='__main__':
    main()
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,datasets,optimizers,models,regularizers

class VGG16(models.Model):

    def  __init__(self,input_shape):
        """
        :param input_shape:[32,32,3]
        """
        super(VGG16,self).__init__()

        weight_decay = 0.000
        self.num_classes = 10

        model = models.Sequential([
            layers.Conv2D(64,(3,3),padding='same',input_shape = input_shape,
            kernel_regularizer=regularizers.l2(weight_decay)),
            layers.Activation('relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),

            

            
        ])

        self.model = model

    def call(self,x):

        x = self.model(x)

        return x
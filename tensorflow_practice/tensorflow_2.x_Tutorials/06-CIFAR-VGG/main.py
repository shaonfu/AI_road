import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,datasets,optimizers
import argparse        #命令行参数解析模块
import numpy as np

from network import VGG16

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
argparse = argparse.ArgumentParser()


argparse.add_argument()
argparse.add_argument()
argparse.add_argument()
argparse.add_argument()

def normalize(X_train,Y_train):
    

    return X_train,Y_train

def prepare_cifar(x,y):

    """
    tf.cast()函数的作用是执行 tensorflow 中张量数据类型转换，
    比如读入的图片如果是int8类型的，一般在要在训练前把图像的数据格式转换为float32。
    """
    x = tf.cast(x,tf.float32)
    y = tf.cast(y,tf.int32)
    return x,y

def compute_loss(logits,labels):
    return tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits,labels=labels
        )
    )

def main():
    
    tf.random.set_seed(22)




if __name__ == '__main__':
    main()
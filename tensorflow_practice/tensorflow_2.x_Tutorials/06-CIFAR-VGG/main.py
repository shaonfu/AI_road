import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,datasets,optimizers
import argparse        #命令行参数解析模块
import numpy as np

from network import VGG16

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
argparse = argparse.ArgumentParser()


argparse.add_argument('--train_dir',type=str,default = '/tmp/cifar10_train',
                        help = "Directory where to write event logs and checkpoint")
argparse.add_argument('--max_steps',type =int, default =1000000,
                        help = "Number of batches to run.")
argparse.add_argument('--log_device_placement',action='store_true',
                        help = "Whether to log device placement.")
argparse.add_argument('--log_frequency',type=int,default = 10,
                        help = "How often to log results to the console")

def normalize(X_train,X_test):
    # this function normalize inputs for zero mean and unit variance
    # it is used when training a model.
    # Input: training set and test set
    # Output: normalized training set and test set according to the trianing set statistics.   
    X_train = X_train / 255. 
    X_test = X_test / 255.

    mean = np.mean(X_train,axis = (0,1,2,3))  #np.mean计算指定轴的算术平均值
    std = np.std(X_train,axis=(0,1,2,3))      #np.std计算指定轴的标准差
    print('mean:',mean,'std:',std)
    X_train = (X_train - mean) / (std + 1e-7)
    X_test = (X_test - mean) / (std + 1e-7)
    return X_train,X_test

def prepare_cifar(x, y):

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

    print('loading data...')
    (x,y),(x_test,y_test) = datasets.cifar10.load_data()
    x,x_test = normalize(x,x_test)
    print(x.shape,y.shape,x_test.shape,y_test.shape)


    train_loader = tf.data.Dataset.from_tensor_slices((x,y))  #是把给定的元组、列表和张量等数据进行特征切片
    train_loader = train_loader.map(prepare_cifar).shuffle(50000).batch(256)

    test_loader = tf.data.Dataset.from_tensor_slices((x_test,y_test))
    test_loader = test_loader.map(prepare_cifar).shuffle(10000).batch(256)
    print('done.')

    model = VGG16([32,32,3])

    #must specify from_logits = True!
    criteon = keras.losses.categorical_crossentropy(from_logits = True)
    metrics = keras.metrics.categorical_accuracy()

    optimizer = keras.optimizers.Adam(learning_rate = 0.0001)

    for epoch in range(250):

        for step,(x,y) in enumerate(train_loader):
            #[b,1] => [b]
            y = tf.squeeze(y, axis = 1)
            #[b,10]
            y = tf.one_hot(y, depth = 10)

            with tf.GradientTape() as tape:
                logits = model(x)
                loss = criteon(y,logits)
                # loss2 = compute_loss(logits, tf.argmax(y, axis=1))
                # mse_loss = tf.reduce_sum(tf.square(y-logits))
                # print(y.shape, logits.shape)
                metrics.update_state(y,logits)

            grads = tape.gradient(loss,model.trainable_variables)
            # MUST clip gradient here or it will disconverge!
            grads = [ tf.clip_by_norm(g, 15) for g in grads]
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            if step % 40 ==0:
                
                print(epoch, step, 'loss:', float(loss), 'acc:', metric.result().numpy())
                metric.reset_states()
        
        if epoch % 1 == 0:

            metric = keras.metrics.CategoricalAccuracy()
            for x, y in test_loader:
                # [b, 1] => [b]
                y = tf.squeeze(y, axis=1)
                # [b, 10]
                y = tf.one_hot(y, depth=10)

                logits = model.predict(x)
                # be careful, these functions can accept y as [b] without warnning.
                metric.update_state(y, logits)
            print('test acc:', metric.result().numpy())
            metric.reset_states()



if __name__ == '__main__':
    main()
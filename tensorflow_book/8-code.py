#%%
#常见网络层类
import tensorflow as tf
#导入keras模型,不能使用import keras,它导入的是标准的keras库
from tensorflow import keras
from tensorflow.keras import layers  #导入常见网络层类

x = tf.constant([2.,1.,0.1])
layer = layers.Softmax(axis=-1)      #创建Softmax层
layer(x)  #调用softmax前向计算

#%%
from tensorflow.keras import layers,Sequential
network = Sequential([          #封杀为一个网络
    layers.Dense(3,activation = tf.nn.relu), #全连接层
    layers.Dense(2,activation = tf.nn.relu)
])
x = tf.random.normal([4,3])
network(x)    #输入从第一层开始,逐层传播至最末层

#%%
#Sequential容器也可以通过add()方法继续追加新的网络层,实现动态创建网络的功能：
import tensorflow as tf
from tensorflow.keras import layers,Sequential

layers_num = 2   #堆叠2次
network = Sequential([])     #先创建空的网络
for _ in range(layers_num):
    network.add(layers.Dense(3,activation = 'relu'))  #添加全连接层
network.build(input_shape=(None,4))   #创建网络参数
network.summary()

#%%
for p in network.trainable_variables:
    print(p.name,p.shape)
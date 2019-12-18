#%%
"""
在TensorFlow中,要实现全连接层，
只需要定义好权值张量W和偏置张量b，
并利用TensorFlow提供的批量矩阵相乘函数
tf.matmul()即可完成网络层的计算。
"""
"""
如下代码创建输入X矩阵为𝑏 = 2个样本，
每个样本的输入特征长度为𝑑𝑖𝑛 = 784，
输出节点数为𝑑𝑜𝑢𝑡 = 256，故定义权值矩阵W的shape 
为[784,256]，并采用正态分布初始化W;
偏置向量 b 的 shape 定义为[256]，
在计算完X@W后相加即可，最终全连接层的输出O的shape
为 [2,256]，即 2 个样本的特征，每个特征长度为 256。
"""
import tensorflow as tf

x = tf.random.normal([2,784])
w1 = tf.Variable(tf.random.truncated_normal([784,256],stddev=0.1))
b1 = tf.Variable(tf.zeros([256]))
o1 = tf.matmul(x,w1) + b1 #线性变换
o1 = tf.nn.relu(o1)   #激活函数

#%%
#层方式实现 高层api
x = tf.random.normal([4,28*28])
from tensorflow.keras import layers #导入层模块
#创建全连接层,指定输出节点数和激活函数
fc = layers.Dense(512,activation=tf.nn.relu)
h1 = fc(x)   #通过fc类完成一次全连接层的运算

# %%
#通过类内部的成员名kernel和bias来获取权值矩阵W和偏置b
print(fc.kernel)
print(fc.bias)

# %%
"""
神经网络
通过堆叠4个全连接层，可以获得层数为4的神经网络，
由于每层均为全连接层，称为全连接网络
"""
"""
在设计全连接网络时，网络的结构配置等超参数
可以按着经验法则自由设置，只需要遵循少量的约束即可。
其中隐藏层1的输入节点数需和数据的实际特征长度匹配，
每层的输入层节点数与上一层输出节点数匹配，
输出层的激活函数和节点数需要根据任务的具体设定进行设计。
总的来说，神经网络结构的自由度较大.
"""
#张量方式实现
#隐藏层1张量
w1 = tf.Variable(tf.random.truncated_normal([784,256],stddev=0.1))
b1 = tf.Variable(tf.zeros([256]))
#隐藏层2张量
w2 = tf.Variable(tf.random.truncated_normal([256,128],stddev=0.1))
b2 = tf.Variable(tf.zeros([128]))
#隐藏层3
w3 = tf.Variable(tf.random.truncated_normal([128,64],stddev=0.1))
b3 = tf.Variable(tf.zeros([64]))
#输出层张量
w4 = tf.Variable(tf.random.truncated_normal([64,10],stddev=0.1))
b4 = tf.Variable(tf.zeros([10]))

with tf.GradientTape() as tape:   #梯度记录器
    # x: [b,28*28]
    # 隐藏层1前向计算,[b,28*28]=>[b,256]
    h1 = x@w1 + tf.broadcast_to(b1, [x.shape[0], 256])
    h1 = tf.nn.relu(h1)
    # 隐藏层2前向计算,[b,256] => [b,128]
    h2 = h1@w2 + b2
    h2 = tf.nn.relu(h2)
    # 隐藏层 3 前向计算，[b, 128] => [b, 64] h3 = h2@w3 + b3
    h3 = tf.nn.relu(h3)
    # 输出层前向计算，[b, 64] => [b, 10] h4 = h3@w4 + b4
    h4 = h3@w4 + b4


#%%
#层方式实现
#新建各个网络层,并指定各层的激活函数类型
fc1 = layers.Dense(256,activation=tf.nn.relu)
fc2 = layers.Dense(128,activation=tf.nn.relu)
fc3 = layers.Dense(64,activation=tf.nn.relu)
fc4 = layers.Dense(10,activation=None)  #输出层

#在前向计算时,依序通过各个网络层即可:
x = tf.random.normal([4,28*28])
h1 = fc1(x)
h2 = fc2(h1)
h3 = fc3(h2)
h4 = fc4(h3)

#通过Sequential容器封装为一个网络类
model = layers.Sequential([
    layers.Dense(256,activation=tf.nn.relu),
    layers.Dense(128,activation=tf.nn.relu),
    layers.Dense(64,activation=tf.nn.relu),
    layers.Dense(10,activation=None)
])

#前向计算时只需要调用一次网络大类对象即可完成所有层的按序计算:
out = model(x)

#%%
#输出层设计
#[0,1]区间,和为1
#softmax函数不仅可以将输出值映射到[0,1]区间
#还满足所有的输出值之和为1的特性
"""
避免单独使用Softmax函数与交叉熵损失函数
下函数将Softmax与交叉熵损失函数同时实现
函数式接口为tf.keras.losses.categorical_crossentropy(y_true, y_pred,from_logits=False)
其中y_true代表了 one-hot 编码后的真实标签,
y_pred表示网络的预测值,当from_logits设置为True时,
y_pred表示须为未经过Softmax函数的变量z;
当from_logits设置为False时,y_pred表示为经过Softmax函数的输出。
"""
z = tf.constant([2.,1.,0.1])
tf.nn.softmax(z)

# %%
z = tf.random.normal([2,10])    #构造输出层的输出
y_onehot = tf.constant([1,3])   #构造真实值
y_onehot = tf.one_hot(y_onehot,depth=10)  #one-hot编码
#输出层未使用Softmax函数,故from_logits设置为True
loss = tf.keras.losses.categorical_crossentropy(y_onehot,z,from_logits=True)
loss = tf.reduce_mean(loss)   #计算平均交叉熵损失
loss

# %%
"""
也可以利用losses.CategoricalCrossentropy(from_logits)类
方式同时实现Softmax与交叉熵损失函数的计算:
"""
#创建Softmax与交叉熵计算类,输出层的输出z未使用softmax
criteon = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
loss = criteon(y_onehot,z)
loss

# %%
#误差计算
"""
常见的误差计算函数有均方差、交叉熵、KL散度、
Hinge Loss函数等,其中均方差函数和交叉熵函数在
深度学习中比较常见，均方差主要用于回归问题，交叉熵主要用于分类问题。
"""
#均方差
#均方差误差(Mean Squared Error, MSE)函数
#把输出向量和真实向量映射到笛卡尔坐标系

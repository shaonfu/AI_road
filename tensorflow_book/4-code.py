#%%
#tensorflow_basic
"""
张量(Tensor) 所有维度数dim > 2的数组统称为张量。
张量的每个维度也做轴(Axis),一般维度代表了具体的物理含义,
比如 Shape 为[2,32,32,3]的张量共有 4 维，如果表 示图片数据的话，
每个维度/轴代表的含义分别是:图片数量、图片高度、图片宽度、 图片通道数，
其中 2 代表了 2 张图片，32 代表了高宽均为 32，3 代表了 RGB 3 个通 道。
张量的维度数以及每个维度所代表的具体物理含义需要由用户自行定义。
"""
import tensorflow as tf
a = 1.2
aa = tf.constant(a)
type(a),type(aa),tf.is_tensor(aa)

# %%
#必须通过TensorFlow规定的方式去创建张量
"""
其中 id 是 TensorFlow 中内部索引对象的编号
shape 表示张量的形状，dtype 表示张量的数值精度，
张量numpy()方法可以返回Numpy.array 类型的数据，
方便导出数据到系统的其他模块:
"""
x = tf.constant([1,2.,3.3])
x

# %%
x.numpy()

# %%
#与标量不同，向量的定义须通过 List 类型传给 tf.constant()。
#创建一个元素的向量
a = tf.constant([1.2])
a,a.shape

# %%
#数值精度
"""
Bit位越长，精度越高，同时占用的内存空间也就越大。
常用的精度类型有tf.int16, tf.int32, tf.int64, 
tf.float16, tf.float32, tf.float64，
其中 tf.float64 即为 tf.double。
"""
tf.constant(123456,dtype = tf.int32)

# %%
#发生溢出
tf.constant(123456,dtype = tf.int16)

#%%
"""
对于大部分深度学习算法，一般使用 tf.int32, tf.float32 
可满足运算精度要求，部分对精度要求较高的算法，
如强化学习，可以选择使用 tf.int64, tf.float64 精度保存张量。
"""
import numpy as np

x = tf.constant(np.pi,dtype = tf.float64)
x
# %%

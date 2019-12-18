"""
TensorFlow进阶
合并与分割,范式统计,张量填充,限幅

"""
#%%
#合并：拼接
import tensorflow as tf

a = tf.random.normal([4,35,8])
b = tf.random.normal([6,35,8])
tf.concat([a,b],axis=0)  #合并成绩册


# %%
#在其他维度上合并张量
a = tf.random.normal([10,35,4])
b = tf.random.normal([10,35,4])
tf.concat([a,b],axis=2)  #在科目维度拼接


# %%
"""
合并操作可以在任意的维度上进行，
唯一的约束是非合并维度的长度必须一致。
比如 shape 为[4,32,8]和 shape 为[6,35,8]的张量
则不能直接在班级维度上进行合并，因为学生数维度的
长度并不一致，一个为 32，另一个为 35:
"""
#非法拼接
a = tf.random.normal([4,32,8])
b = tf.random.normal([6,35,8])
tf.concat([a,b],axis=0)  #非法拼接

# %%

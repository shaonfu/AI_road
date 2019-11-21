#%%
import numpy as np
print(np.__version__)

#%%
L = list(range(10))
L

#%%
type(L[0])

#%%
L2 = [str(c) for c in L]
L2

#%%
type(L2[0])

#%%
L3 = [True,'s',12,12.0]
[type(item) for item in L3]

#%%
import numpy as np
#integer array
np.array([1,4,2,5,3])

#%%
#创建一个3*3的、在0~1均匀分布的随机数组成的数组
np.random.random((3,3))

#%%
#创建一个3*3的、均值为0，方差为1的
#正态分布的随机数数组
np.random.normal(0,1,(3,3))

#%%
#创建一个3*3的、[0，10)区间的随机整型数组
np.random.randint(0,10,(3,3))


#%%
#创建一个单位矩阵
np.eye(3)

#%%
np.empty(3)

#%%
np.random.seed(0) #设置随机种子

x1 = np.random.randint(10,size = 6)  #一维数组
x2 = np.random.randint(10,size =(3, 4)) #二维数组
x3 = np.random.randint(10,size =(3,4,5))   #三维数组

print("x3 ndim: ",x3.ndim)   #数组的维度
print("x3 shape: ",x3.shape)   #数组每个维度的大小
print("x3 size: ",x3.size)  #数组的总大小
#%%
print("x3 dtype:",x3.dtype)  #数组的数据类型

#%%
x1

#%%
x2

#%%
x3

#%%
x2[-1,-4]

#%%
#数组切片
x = np.arange(10)
x

#%%
x[:5]  #前五个元素


#%%
x[5:]  #索引后五个元素

#%%
x[4:7]   #中间的子数组 前开后闭

#%%
x[::2]   #每隔一个元素

#%%
x[1::2]  #每隔一个元素 从索引1开始

#%%
x[::-1]  #逆序数组

#%%
x[5::-2] #从索引5开始每隔一个元素逆序

#%%
x2

#%%
#多维度数组切片
x2[:2,:3]  #两行，三列

#%%
x2[:3,::2]  #所有行,每隔一列

#%%
#获取数组的行和列
print(x2[:,0])  #x2的第一列

#%%
print(x2[0,:])  #x2的第一行

#%%
x2_sub = x2[:2,:2]
print(x2_sub)
#%%
x2_sub[0,0] = 99
print(x2_sub)
print(x2)

#%%
#数组的变形
#将数字1~9放入3*3的矩阵
grid = np.arange(0,9).reshape(3,3)

#%%
print(grid)

#%%
#聚合
x = np.arange(1,6)

#返回数组内所有元素的和
np.add.reduce(x)

#%%
#返回数组内所有元素的乘积
np.multiply.reduce(x)

#%%
#如果需要存储每次计算得到中间结果,可以使用accumulate
np.add.accumulate(x)

#%%
np.multiply.accumulate(x)

#%%
#获得两个不同输入数组所有元素对的函数运算结果
y = np.arange(1,7)
np.multiply.outer(x,y)

#%%
L = np.random.random(100)
sum(L)

#%%
np.sum(L)

#%%
big_array = np.random.random(1000000)
%timeit sum(big_array)
%timeit np.sum(big_array)

#%%
min(big_array),max(big_array)

#%%
np.min(big_array),np.max(big_array)

#%%
%timeit max(big_array)
%timeit np.max(big_array)

#%%
#多维度聚合
M = np.random.random((3,4))
print(M)

#%%
#默认返回对整个数组的聚合结果
M.sum()

#%%
#用于指定沿着哪个轴的方向进行聚合
#axis关键字指定的是数组将会被折叠的维度
#对于一个二维空间，axis=1代表横轴，axis=0按照竖轴
#输出结果对应四列数字的计算值
M.min(axis = 0)

#%%
#每一行的最大值
M.max(axis = 1)

#%%
#创建一个5个元素的数组,这5个数均匀地分配到0~1
np.linspace(0,1,5)


#%%
"""
Numpy广播
1.如果两个数组的维度数不相同,那么小维度数组的形状将会在最左边补1
2.如果两个数组的形状在任何一个维度上都不匹配,那么数组的形状会沿着
维度为1的维度扩展以匹配另外一个数组的形状
3.如果两个数组的形状在任何一个维度上都不匹配且没有任何一个维度等于1,那么会引发异常
"""
#广播示例1
#将一个二维数组与一个一维数组相加
M = np.ones((2,3))
a = np.arange(3)

print(M)
print(a)

# %%
print(M.shape)
print(a.shape)

# %%
#a数组维度左边补1,一个维度不匹配扩展这个维度以匹配数组,同时变为（2,3）
M + a

# %%
a = np.arange(3).reshape((3,1))
b = np.arange(3)

print(a.shape)
print(b.shape)

# %%
#根据规则1 用1将b的形状补全 b.shape=(1,3)
#根据规则2 更新两个数组的维度来互相匹配 因此ab数组均为（3，3）
print(a)
print(b)

a+b

# %%
c = np.ones((3,1))
print(c)


# %%
#广播示例3
#这是一个不兼容的数组配置
M = np.ones((3,2))
a = np.arange(3)

#%%
#广播的实际应用
#数组的归一化
X = np.random.random((10,3))

# %%
print(X)

# %%
#计算方法是利用mean函数沿着第一个维度聚合
Xmean = X.mean(0)
Xmean

# %%
#通过从X数组的元素中减去这个均值实现归一化（这是一个广播操作）
X_centered = X - Xmean

# %%
#检查归一化之后的数组的均值是否接近0
X_centered.mean(0)

# %%
#数组的排序
#一个简单的选择排序,重复寻找列表的最小值
import numpy as np

def selection_sort(x):
    for i in range(len(x)):
        swap = i+np.argmin(x[i:])
        (x[i],x[swap]) = (x[swap],x[i])
    return x

x = np.array([2,1,4,3,5])
selection_sort(x)


# %%
#快速排序np.sort和np.argsort
x = np.array([2,1,4,5,8])
np.sort(x)


# %%
#希望用排好序的数组替代原始数组,可以使用数组的sort方法:
x.sort()
print(x)

# %%
#argsort,该函数返回的是原始数组排好序的索引值
x = np.array([6,1,4,9,3])
i = x.argsort()
print(i)

# %%
#这些索引值可以被用于（通过花哨的索引）创建有序的数组
x[i]

# %%n
rand = np.random.RandomState(42)
X = rand.randint(0,10,(4,6))
print(X)

# %%
#对X的每一列排序
np.sort(X,axis=0)

# %%
#对X的每一行排序
np.sort(X,axis=1)

# %%
#部分排序:分隔 np.partition,输入数组与数字K
#输出结果是一个新数组,最左边是第K小的值
x = np.array([7,2,3,1,6,5,4])
np.partition(x,3)


# %%
#示例:K个最近邻
X = rand.rand(10,2)

# %%
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn;
seaborn.set()
plt.scatter(X[:,0],X[:,1],s=100)

# %%

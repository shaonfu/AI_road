#%%
import matplotlib
matplotlib.use('MacOSX')
import matplotlib.pyplot as plt

#%%
plt.style.use('classic')

#%%
import numpy as np
x = np.linspace(0,10,100)
fig = plt.figure()
plt.plot(x,np.sin(x),'_')
plt.plot(x,np.cos(x),'_')
plt.show()

#%%
#查看系统支持的文件格式
fig.canvas.get_supported_filetypes()

#%%
#面向对象接口
#先创建图形网络
#ax是一个包含两个Axes对象的数组
fig,ax = plt.subplots(2)
ax[0].plot(x,np.sin(x))
ax[1].plot(x,np.cos(x))

#%%
#MATLAB风格接口
plt.figure()  #创建图形

#创建两个子图中的第一个,设置坐标轴
plt.subplot(2,1,1)  #行,列,子图编号
plt.plot(x, np.sin(x))

#创建两个子图中的第二个,设置坐标轴
plt.subplot(2,1,2)  #行,列,子图编号
plt.plot(x, np.cos(x))

#%%
# 获取所有的自带样式
print (plt.style.available)

#%%
plt.style.use("seaborn-whitegrid")
import numpy as np


#%%
#一个空的网格坐标轴
#先创建一个图形fig和一个坐标轴ax
fig = plt.figure()
ax = plt.axes()

#%%
x = np.linspace(0,10,1000)
ax.plot(x,np.sin(x))

#%%
#pylab接口画图
plt.plot(x, np.sin(x))

#%%
plt.plot(x, np.cos(x))
plt.plot(x, np.sin(x))

#%%
plt.plot(x, np.sin(x),color='red')
plt.plot(x, np.sin(x-1),color='green')
plt.plot(x, np.sin(x-2),color='0.75')    #灰度值

#%%
import numpy as np
x = np.linspace(0,10,1000)
plt.plot(x,x+0,linestyle ='solid')  #实线
plt.plot(x,x+1,linestyle ='--')    #虚线
plt.plot(x,x+2,linestyle ='-.')     #点划线
plt.plot(x,x+3,linestyle =':')      #实点线

#%%
plt.plot(x,x+0,'-g')  #绿色实线
plt.plot(x,x+1,'--c')    #青色虚线
plt.plot(x,x+2,'-.k')     #点黑色划线
plt.plot(x,x+3,':r')      #红色实点线

#%%
#自定义坐标轴上下限
plt.plot(x,np.sin(x))

plt.xlim(-1,11)
plt.ylim(-1.5,1.5)

#%%
#axis()可以按照图形的内容自动收紧坐标轴
plt.plot(x,np.sin(x))
plt.axis('tight')

#%%
plt.plot(x,np.sin(x))
plt.axis('equal')

#%%
#图形标签
plt.plot(x,np.sin(x))
plt.title("A Sine Curve")
plt.xlabel("x")
plt.ylabel("sin(x)")

#%%
#图例
plt.plot(x,np.sin(x),':b',label='sin(x)')
plt.plot(x,np.cos(x),'-g',label='cos(x)')
plt.axis("equal")

plt.legend()

#%%
#散点图
x = np.linspace(0,10,30)
y = np.sin(x)

plt.plot(x,y,'o',color = 'black')

#%%
#散点图
rng = np.random.RandomState(0)
for marker in ['o','.',',','x','+','v','^','<','>','s','d']:
    plt.plot(rng.rand(5),rng.rand(5),marker,label="marker='{0}'".format(marker))
    plt.legend(numpoints=1)
    plt.xlim(0,1.8)

#%%
#连线
#plt.plot函数非常灵活 可以满足不同的可视化配置需求
plt.plot(x,y,'-ok')   #直线（-）,圆圈（o）,黑色（k）

#%%
#另一个散点图
plt.scatter(x,y,marker = 'o')

#%%
rng = np.random.RandomState(0)
x = rng.randn(100)
y = rng.randn(100)
colors = rng.rand(100)
sizes = 1000*rng.rand(100)

#%%
plt.scatter(x, y, c = colors, s = sizes, alpha =0.3,cmap ='viridis')
plt.colorbar()  #显示颜色条

#%%
#plt.plot与plt.scatter
#在面对大型数据集的时候,plot方法更加好
from sklearn.datasets import load_iris
iris = load_iris()
features = iris.data.T

plt.scatter(features[0], features[1], 
            c=iris.target, s=100*features[3], alpha=0.2,cmap = "viridis")

plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])


#%%
#基本误差线
x = np.linspace(0,10,50)
dy = 0.8
y = np.sin(x) + dy * np.random.randn(50)

plt.errorbar(x,y,yerr = dy, fmt ='.k') 


#%%
#误差线的参数
plt.errorbar(x, y, yerr=dy, fmt="o",color = 'black',
             ecolor='lightgray',elinewidth=3,capsize=0)

#%%
#连续误差
from sklearn.gaussian_process import GaussianProcess

#定义模型和要画的数据
model = lambda x: x * np.sin(x)
xdata = np.array([1,3,5,6,8])
ydata = model(xdata)

#计算高斯过程拟合结果
gp = GaussianProcess(corr='cubic',theta0 = 1e-2,thetal=1e-4,thetaU=1E-1,
                     random_start=100)
gp.fit(xdata[:,np.newaxis],ydata)

xfit = np.linspace(0,10,1000)
yfit,MSE = gp.predict(xfit[:,np.newaxis],eval_MSE=True)
dyfit = 2*np.sqrt(MSE)       #2*sigma~95%置信区间


#%%
#密度线与等高线
plt.style.use('seaborn-white')
def f(x,y):
    return np.sin(x) ** 10 + np.cos(10+y*x)*np.cos(x)

#%%
x = np.linspace(0,5,50)
y = np.linspace(0,5,40)

X,Y = np.meshgrid(x,y)
Z = f(X,Y)

#%%
plt.contour(X,Y,Z,colors = 'black')

#%%
#配置图例
plt.style.use("classic")

x = np.linspace(0,10,1000)
fig,ax = plt.subplots()
ax.plot(x,np.sin(x),'-b',label='Sine')
ax.plot(x,np.cos(x), "--r", label = "Cosine")
ax.axis('equal')
leg = ax.legend()

#%%
#设置图例的位置,并取消外边框
ax.legend(loc = 'upper left',frameon = False)
fig

#%%
#ncol设置标签列数
ax.legend(loc = 'lower center',frameon = False,ncol = 2)
fig

#%%
#定义圆角边框(fancybox)、增加阴影、改变外边框透明度(framealpha值),或者改变文字间距
#关于图例更多配置信息,参考plt.legend程序文档
ax.legend(fancybox = True,framealpha =1,shadow = True,borderpad =1)
fig
#%%
y = np.sin(x[:,np.newaxis]+np.pi*np.arange(0,2,0.5))
lines = plt.plot(x,y)

#lines变量是一组plt.Line2D实例
plt.legend(lines[:2],['first','second'])

#%%
plt.plot(x,y[:,0],label='first')
plt.plot(x,y[:,1],label='second')
plt.plot(x,y[:,2:])
plt.legend(framealpha=1,frameon = True)

#%%
#案例:手写数字
#加载数字0~5的图形,对其进行可视化
from sklearn.datasets import load_digits
digits = load_digits(n_class=6)

#%%
fig,ax = plt.subplots(8,8,figsize=(6,6))
for i,axi in enumerate(ax.flat):
    axi.imshow(digits.images[i],cmap='binary')
    axi.set(xticks=[],yticks=[])

#%%
#用IsoMap方法将数字投影到二维空间
from sklearn.manifold import Isomap
iso = Isomap(n_components=2)
projection = iso.fit_transform(digits.data)

#%%
#画图
plt.scatter(projection[:,0],projection[:,1],lw=0.1,
c=digits.target,cmap=plt.cm.get_cmap('cubehelix',6))
plt.colorbar(ticks=range(6),label='digit value')
plt.clim(-0.5,5.5)
plt.show()
#%%

#%%

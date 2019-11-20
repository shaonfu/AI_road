#%%
import numpy as np
# %%
from __future__ import  print_function
import torch

# %%
x = torch.empty(5,4)
print(x)

# %%
from fastai import *
from fastai.vision import *

#%%

from IPython import get_ipython 
#%%
path_img = ".\datas\cat_dog_images"
  
#%%
fnames = get_image_files(path_img)

#%%
fnames[:5]

#%%
#fastai库提供了一个非常好用的函数来实现将文件名中标签信息提取出来
#ImageDataBunch.from_name_re函数通过使用正则表达式 从文件名中提取标签信息
np.random.seed(2)
pat = r'/([^/]+)_\d+.jpg$'

#%%
bs = 25 # batch_size

#%%
data = ImageDataBunch.f rom_name_re(path_img,fnames,pat,ds_tfms=get_transforms(),size=224,bs=bs,
                                   num_workers=0).normalize(imagenet_stats)

#%%
data.show_batch(rows=3,figsize=(7,6))

#%%
print(data.classes) 

# %%
len(data.classes),data.c

#%%
learn = cnn_learner(data,models.resnet34,metrics=error_rate)

#%%
learn.model

# %%
learn.fit_one_cycle(5)

#%%
#保存路径与数据来源目录一致
learn.save('stage-1')

#%%
learn = learn.load("stage-1")

#%%
interp = ClassificationInterpretation.from_learner(learn)
w
# %%
interp.plot_top_losses(9,figsize=(15,11))

#%%
doc(interp.plot_top_losses)

#%%
#混淆矩阵
interp.plot_confusion_matrix(figsize=(12,12),dpi=60)

# %%
interp.most_confused(min_val=2)

# %%
learn.lr_find()

# %%
learn.recorder.plot()

# %%
learn.unfreeze()
learn.fit_one_cycle(2,max_lr = slice(1e-4,1e-2))

# %%
import torch
a = torch.cuda.is_available()
print(a)

ngpu= 1
# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
print(device)
print(torch.cuda.get_device_name(0))
print(torch.rand(3,3).cuda()) 



# %%
torch.cuda.is_available()
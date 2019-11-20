#%% 
from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())
#%% 

#%% 
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

#%% 
value = tf.test.is_gpu_available(
    cuda_only=False,
    min_cuda_compute_capability=None
)
print ('***If TF can access GPU: ***\n\n',value) # MUST RETURN True IF IT CAN!!
#%% 

tf.test.is_gpu_available()

#%%
import tensorflow as tf

print(tf.__version__)

# %%

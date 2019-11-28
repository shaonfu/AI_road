import tensorflow as tf
from tensorflow.keras import dattasets,layers,optimizers,Sequential,metrics

(xs,ys),_ = datasets.mnist.load_data()
print('datasets:',xs.shape,ys.shape,xs.min(),xs.max())

xs = tf.convert_to_tensor(xs,dtype = tf.float32) / 255.
db = tf.data.Dataset.from_tensor_slices((xs,ys))
db = db.batch(32).repeat(10)

model = Sequential([
    layers.Dense(256,activation ='relu'),
    layers.Dense(256,activation ='relu'),
    layers.Dense(256,activation ='relu'),
    layers.Dense(10)
])
model.build(input_shape=(None,28*28))
model.summary()

optimizer = optimizers.SGD(lr=0.01)
acc_meter = metrics.Accuracy()



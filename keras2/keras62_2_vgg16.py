import numpy as np
# from tensorflow.python.keras.models import Sequential
# from tensorflow.python.keras.layers import Dense,Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten
from keras.datasets import cifar10
import tensorflow as tf

tf.random.set_seed(777)
np.random.seed(777)
print(tf.__version__)

from keras.applications import VGG16

vgg16 = VGG16(weights='imagenet',include_top=False,input_shape=(32,32,3))
vgg16.trainable = False # 가중치 동결

model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(10,activation='softmax'))

# model.build((None, 32, 32, 3))

model.summary()


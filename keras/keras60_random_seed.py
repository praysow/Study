import tensorflow as tf
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense
import random as rn
print(np.__version__)
print(tf.__version__)
print(keras.__version__)
rn.seed(333)
tf.random.set_seed(123)
np.random.seed(321)
x= np.array([1,2,3])
y= np.array([1,2,3])

model=Sequential()
model.add(Dense(5,input_dim=1))
model.add(Dense(5))
model.add(Dense(1))

model.compile(loss='mse',optimizer='adam')
model.fit(x,y,epochs=100)

loss= model.evaluate(x,y)
result=model.predict([4])
print("L",loss)
print("R",result)
'''
L 0.00010596482752589509
R [[3.9783478]]
L 6.554233551025391
R [[-0.9801741]]
'''
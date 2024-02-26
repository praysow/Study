import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import tensorflow as tf

tf.random.set_seed(777)
np.random.seed(777)
print(tf.__version__)

x = np.array([1,2])
y = np.array([1,2])

model = Sequential()
model.add(Dense(2,input_dim = 1))
# model.add(Dense(2))
model.add(Dense(1))

# model.summary()
print(model.weights)
model.trainable = False
print("===============================================================")

model.compile(loss= 'mse',optimizer='adam')
model.fit(x,y,batch_size=10,epochs=0)

y_pred = model.predict(x)
print(y_pred)
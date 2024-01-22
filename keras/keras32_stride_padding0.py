import numpy as np
import pandas as pd
from keras.datasets import mnist
from keras.layers import Dense, Conv2D, Flatten     #Flatten: 평탄화시키다
from keras.models import Sequential
from keras.utils import to_categorical
import time
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

model = Sequential()
model.add(Conv2D(1, (2, 2), input_shape=(28, 28, 1),strides=2,padding='same')) #padding same하면 사이즈 그대로 유지된다 1은 한줄추가 2는 전체추가
model.add(Conv2D(2, (3, 3)))
model.add(Conv2D(3, (4, 4)))
model.add(Flatten())
model.add(Dense(20))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10, activation='softmax'))

model.summary()

# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# start_time = time.time()
# model.fit(x_train, y_train, epochs=100, batch_size=32, verbose=1, validation_split=0.2)
# end_time = time.time()

# result = model.evaluate(x_test, y_test)
# print("loss", result[0])
# print("acc", result[1])
# print("걸린시간 :",round(end_time - start_time))



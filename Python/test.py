from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import tensorflow as tf
import keras
print("tf 버전: ",tf.__version__)
print("keras 버전 :", keras.__version__)

#1. 데이터
x = np.array([[1,2,3,4,5,6],[7,8,9,10,11,12]])
y = np.array([[1,2,3,5,4,6],[7,8,9,10,11,12]])


#2.모델구성
model=Sequential()
model.add(Dense(10, input_dim=6))
model.add(Dense(70))
model.add(Dense(100))
model.add(Dense(70))
model.add(Dense(50))
model.add(Dense(1))

#3.컴파일 훈련
model.compile(loss='mse',optimizer='adam')
model.fit(x, y, epochs=100, batch_size=2)                   #배치사이즈 : 데이터를 하나씩 나눠서 작업하겠다, 훈련량 늘리기, 숫자가 작을수록 여러번 작업

#4.결과예측
loss=model.evaluate(x,y)
result = model.predict([[1,2,3,4,5,6]])
print("로스 :", loss)
print("7의 예측값 :", result)


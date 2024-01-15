from keras.models import Sequential
from keras.layers import Dense
import numpy as np

#1. 데이터
x=np.array([1,2,3])
y=np.array([1,2,3])


#2모델구성
model= Sequential()
model.add(Dense(5,input_shape=(1,)))
model.add(Dense(4))
model.add(Dense(2))
model.add(Dense(1))

model.summary()

#3.컴파일 훈련






#4.결과예측
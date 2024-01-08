#06.1 카피

from keras.models import Sequential
from keras.layers import Dense
import numpy as np
#1. 데이터
# x=np.array([1,2,3,4,5,6,7,8,9,10])
# y=np.array([1,2,3,4,5,6,7,8,9,10])

x_train=np.array([1,2,3,4,5,6,7])
y_train=np.array([1,2,3,4,6,5,7])

x_val = np.array([6,7])
y_val = np.array([x,y])
x_test = np.array([8,9,10])
y_test = np.array([8,9,10])

#2. 모델구성
model=Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(70))
model.add(Dense(100))
model.add(Dense(70))
model.add(Dense(50))
model.add(Dense(1))

#3.컴파일 훈련
model.compile(loss='mse',optimizer='adam')
model.fit(x_train, y_train, epochs=510, batch_size=1,validation_data=(x_val,y_val))                


#4.결과예측
loss=model.evaluate(x_test,y_test)
result = model.predict([7,11000000])
print("로스 :", loss)
print("8,9,10의 예측값 :", result)
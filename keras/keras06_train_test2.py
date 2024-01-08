from keras.models import Sequential
from keras.layers import Dense
import numpy as np
#1. 데이터
x=np.array([1,2,3,4,5,6,7,8,9,10])
y=np.array([1,2,3,4,5,6,7,8,9,10])
#넘파이 리스트의 슬라이싱 7:3으로 잘라라
x_train= x[:7]
y_train= y[:7]
x_test = x[7:]
y_test = y[7:]
# print(x_train)
# print(y_train)
# print(x_test)
# print(y_test)
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
model.fit(x_train, y_train, epochs=510, batch_size=1)                   #배치사이즈 : 데이터를 하나씩 나눠서 작업하겠다, 훈련량 늘리기, 숫자가 작을수록 여러번 작업
#4.결과예측
loss=model.evaluate(x_test,y_test)
result = model.predict([7,11000000])
print("로스 :", loss)
print("7,11000000의 예측값 :", result)
# 로스 : 3.031649096259942e-13
# 7,11000000의 예측값 : [[7.0000000e+00]                    10,70,100,70,50,1,   epochs=510, batch_size=1
#  [1.0999999e+07]]
from keras.models import Sequential
from keras.layers import Dense
import numpy as np

#1. 데이터
x = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
             [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.5, 1.4, 1.3],
             [9,8,7,6,5,4,3,2,1,0],])
y = np.array([1,2,3,4,5,6,7,8,9,10])

x=x.T
# print(x)
# print(x.shape, y.shape)

#2. 모델구성

model=Sequential()
model.add(Dense(8,input_dim=3))
model.add(Dense(100))
model.add(Dense(80))
model.add(Dense(60))
model.add(Dense(40))
model.add(Dense(20))
model.add(Dense(1))


#3. 컴파일, 훈련

model.compile(loss='mse',optimizer='adam')
model.fit(x,y, epochs=100, batch_size=3)

#4. 결과예측
loss=model.evaluate(x,y)
result=model.predict([[10,1.3,0]])
print("로스 :",loss)
print("[10,1.3,0]의 예측값 : ",result)


# 로스 : 0.0009193670703098178                          8,100,80,60,40,20,1/ epochs=100 batch_size=3
# [10,1.3,0]의 예측값 :  [[10.001851]]
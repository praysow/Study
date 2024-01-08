from keras.models import Sequential
from keras.layers import Dense
import numpy as np
#1.
x=np.array([1,2,3,4,5,6,7,8,9,10])
y=np.array([1,2,3,4,6,5,7,8,9,10])
#2.
model=Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(70))
model.add(Dense(100))
model.add(Dense(70))
model.add(Dense(50))
model.add(Dense(1))

#3.컴파일 훈련
model.compile(loss='mse',optimizer='adam')
model.fit(x, y, epochs=510, batch_size=1)                   #배치사이즈 : 데이터를 하나씩 나눠서 작업하겠다, 훈련량 늘리기, 숫자가 작을수록 여러번 작업

#4.결과예측
loss=model.evaluate(x,y)
result = model.predict([11000000, 7])
print("로스 :", loss)
print("7의 예측값 :", result)


# 로스 : 9.96180893851617e-13               10,70,100,70,50,1,epochs=510, batch_size=1
# 7의 예측값 : [[11000000.]]

# 로스 : 6.252776074688882e-13              10,70,100,70,50,1,epochs=510, batch_size=1
# 7의 예측값 : [[1.0999999e+07]
#  [7.0000000e+00]]

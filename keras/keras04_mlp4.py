import numpy as np
from keras.models import Sequential
from keras.layers import Dense

#1. 데이터
x= np.array(range(10))     #range(10)= 0~9 range는 이미[]를 포함하고 있는 단어이다  (10,)

y= np.array([[1,2,3,4,5,6,7,8,9,10],[1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9],[9,8,7,6,5,4,3,2,1,0]])  #[]는 리스트라고 한다    (3,10)
y= y.transpose()

#2. 모델구성
model=Sequential()
model.add(Dense(1,input_dim=1))
model.add(Dense(300))
model.add(Dense(200))
model.add(Dense(100))
model.add(Dense(60))
model.add(Dense(30))
model.add(Dense(3))


#3. 컴파일, 훈련
model.compile(loss="mse",optimizer='adam')
model.fit(x,y, epochs=300, batch_size=1)

#4. 결과예측
loss=model.evaluate(x,y)
result=model.predict([11000000])
print("로스 :",loss)
print("11,2,-1 :", result)


# 로스 : 6.452040111071256e-07
# 11,2,-1 : [[11.000063   1.9998192 -1.0016667]]          1,300,200,100,60,30,3     epochs=300, batch_size=1
import numpy as np
from keras.models import Sequential
from keras.layers import Dense

#1. 데이터
x= np.array([range(10)])                    #range는 파이썬에서 기본으로 제공하는 함수
print(x)          #[[0 1 2 3 4 5 6 7 8 9]]   #전산에서 숫자는 0부터 시작(0~9) 10은 10-1=9로 읽는다
print(x.shape)    #(1, 10)

x=np.array([range(10),range(21,31), range(201,211)])

                            #range는 파이썬에서 기본으로 제공하는 함수
print(x)         #[[1 2 3 4 5 6 7 8 9]]        #전산에서 숫자는 0부터 시작(0~9) 10은 10-1=9로 읽는다
print(x.shape)   #(1, 9)
print(x)
print(x.shape)
x= x.transpose()
print(x)
print(x.shape)

y= np.array([[1,2,3,4,5,6,7,8,9,10],[1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9],[9,8,7,6,5,4,3,2,1,0]])  #[]는 리스트라고 한다
y=y.transpose()

#2.모델구성
model=Sequential()
model.add(Dense(50,input_dim=3))
model.add(Dense(90))
model.add(Dense(100))
model.add(Dense(80))
model.add(Dense(20))
model.add(Dense(3))

#3.컴파일 훈련

model.compile(loss='mse',optimizer='adam')
model.fit(x,y, epochs=100, batch_size=1)


#4.결과예측
loss=model.evaluate(x,y)
result=model.predict([[10,31,211]])
print("로스",loss)
print("10,31,211의 예측값", result)

# 로스 3.145704496887447e-08                                                50,90,100,80,20,3 epochs=100, batch_size=1
# 10,31,211의 예측값 [[11.00001     2.000468   -0.99969625]]




import numpy as np
from keras.models import Sequential
from keras.layers import Dense

#1. 데이터
x = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
             [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.5, 1.4, 1.3]])
y = np.array([1,2,3,4,5,6,7,8,9,10])

print(x.shape)          #(2, 10)          데이터확인은 열(속성)
print(y.shape)          #(10,)
# [[1,1],[2,1.1],[3,1.2],[4,1.3]]
x = x.transpose()       
# x=x.T 는 x = x.transpose() 와 같다 
print(x.shape)

#2. 모델구성
model=Sequential()
model.add(Dense(5, input_dim=2)) #열, 컬럼, 속성, 특성, 차원 =2 // 같다. (dim은 열(벡터)의 개수)
                                # (행 무시, 열 우선) <= 외우기,  벡터와 input_dim 행렬은 똑같이 한다
model.add(Dense(100))
model.add(Dense(70))
model.add(Dense(50))
model.add(Dense(30))
model.add(Dense(1))

# 3. 컴파일,훈련

model.compile(loss='mse',optimizer='adam')
model.fit(x,y, epochs=100,batch_size=3)

#4. 결과에측
loss=model.evaluate(x,y)
result = model.predict([[10,1.3]])
print("로스 :", loss)
print("11의 예측값:",result)

# 로스 : 2.459689767420059e-06
# [[10, 1.3]]의 예측값: [[10.001245]]       5,100,70,50,30,1, epoch=100, batch=3
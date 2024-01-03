import tensorflow as tf # tensorflow를 땡겨오고, tf라고 줄여서 쓴다.
print(tf.__version__)   # 2.15.0
from tensorflow.keras.models import Sequential #순차적모델
from tensorflow.keras.layers import Dense      #군집
import numpy as np

#1. 데이터
x = np.array([1,2,3])       #numpy데이터에 1,2,3을 추가한다
y = np.array([1,2,3])

#2. 모델구성
model = Sequential()                    #순차적 모델을 만드는데 x라는 데이터 한덩어리를 1이라고 한다
model.add(Dense(1, input_dim=1))        #input = x, x의 데이터 한덩어리 dim=차원(데이터 덩어리), x한개의 차원 y한개의 차원

#3. 컴파일, 훈련
model.compile(loss='mse' , optimizer='adam')       #mse는 절대값이 아닌 제곱하는 방식으로 양수로 만들겠다
model.fit(x,y, epochs=10000)       #최적의 weight생성,fit= 훈련 epochs=훈련양을 조절
                                #weight 값에서 loss값을 뺌
#4. 평가, 예측
loss = model.evaluate(x,y)
print(" 로스 : ", loss)
result = model.predict([4])
print("4의 예측값은 : ", result)


#파이썬은 인터프린트언어다
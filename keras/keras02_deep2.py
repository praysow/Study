import numpy as np
from keras.models import Sequential
from keras.layers import Dense

#1. 데이터
x = np.array([1,2,3,4,5,6])
y = np.array([1,2,3,5,4,6])

#2 모델구성
model = Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(500))
model.add(Dense(1000))
model.add(Dense(500))
model.add(Dense(3))
model.add(Dense(1))
      

#3. 컴파일, 훈련
model.compile(loss='mse' , optimizer='adam')       
model.fit(x,y, epochs=100, batch_size=3)      
                               
#4. 평가, 예측
loss = model.evaluate(x,y)
print(" 로스 : ", loss)
result = model.predict([1,2,3,4,5,6,7])
print("7의 예측값은 : ", result)

# 로스 :  0.3270958960056305
# 1/1 [==============================] - 0s 71ms/step
# 7의 예측값은 :  [[1.0487748]
#  [2.0244515]
#  [3.0001285]
#  [3.9758055]
#  [4.9514823]
#  [5.9271593]
#  [6.902837 ]]
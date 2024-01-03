# from keras.models import Sequential
# from keras.layers import Dense
import numpy as np
from keras.models import Sequential
from keras.layers import Dense

#1. 데이터
x = np.array([1,2,3,4,5,6])
y = np.array([1,2,3,5,4,6])

#2
model = Sequential()                    
model.add(Dense(1, input_dim=1))        

#3. 컴파일, 훈련
model.compile(loss='mse' , optimizer='adam')       
model.fit(x,y, epochs=10000)      
                               
#4. 평가, 예측
loss = model.evaluate(x,y)
print(" 로스 : ", loss)
result = model.predict([1,2,3,4,5,6,7])
print("7의 예측값은 : ", result)


# 로스 :  0.3238094747066498
# 1/1 [==============================] - 0s 47ms/step
# 7의 예측값은 :  [[1.1428571]
#  [2.0857143]
#  [3.0285714]
#  [4.9142857]
#  [3.9714286]
#  [5.857143 ]
#  [6.8      ]
#epochs = 6000
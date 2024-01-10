from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import r2_score, mean_squared_error                    #케라스에서는 훈련데이터를 일부 짤라서 validation으로 사용 가능
import numpy as np
import pandas as pd
#1.데이터
x=np.array(range(1,17))
y=np.array(range(1,17))

x_train,x_test,y_train,y_test=train_test_split(x,y, train_size=0.85 , random_state=10)




# x_train,x_val,y_train,y_val=train_test_split(x,y, train_size=0.85, random_state=10)


# print(x_train,y_train)
# print(x_test,y_test)
# print(x_val,y_val)
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
model.fit(x_train, y_train, epochs=100, batch_size=1,validation_split=0.3,verbose=2)                


#4.결과예측
loss=model.evaluate(x_test,y_test)
result = model.predict([x_test])
print("로스 :", loss)
print("예측값 :", result)








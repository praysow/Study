from keras.models import Sequential
from keras.layers import Dense
import numpy as np
#1. 데이터

x=np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
y=np.array([1,2,4,3,5,7,9,3,8,12,13,8,  14,15,9, 6,17,23,21,20])

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test =train_test_split(x,y, test_size=0.1,random_state=450)

# print(x_train)
# print(y_train)
# print(x_test)
# print(y_test)

#2.모델구성
model=Sequential()
model.add(Dense(50,input_dim=1))
model.add(Dense(80))
model.add(Dense(120))
model.add(Dense(90))
model.add(Dense(30))
model.add(Dense(1))

#3.컴파일, 훈련
model.compile(loss='mse',optimizer='adam')
model.fit(x_train,y_train,epochs=510, batch_size=1)

#4.결과 예측
loss=model.evaluate(x_test,y_test)
y_predict=model.predict([x_test])
result=model.predict(x)
print("로스 :",loss)
# print("x 예측값",result)

import matplotlib.pyplot as plt

from sklearn.metrics import r2_score
r2=r2_score(y_test,y_predict)
print("R2 score",r2)

# plt.scatter(x,y)
# # plt.plot(x, y_predict,color='red')
# plt.scatter(x, result,color='red')
# plt.show()
'''

로스 : 1.2174264192581177             random_state=450
R2 score 0.9745337895539326         50,80,120,90,30,1,epochs=510, batch_size=2
로스 : 0.7247693538665771
R2 score 0.989060084843082  
로스 : 0.3976149260997772           test_size=0.1,random_state=450
R2 score 0.9937872669870983         50,80,120,90,30,1,epochs=510, batch_size=2
'''
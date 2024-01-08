#1.R2를 음수가 아닌0.5 미만로 만들것
#2.데이터는 건들지 말것
#3.레이어는 인풋과 아웃풋을 포함해서 7개 이상
#4.batch_size+1
#5. 히든레이어는 노드는 10개 이상 100개 이하
#6. train 사이즈 75%
#7. epoch 100번 이상


from keras.models import Sequential
from keras.layers import Dense
import numpy as np
#1. 데이터

x=np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
y=np.array([1,2,4,3,5,7,9,3,8,12,13,8,  14,15,9, 6,17,23,21,20])

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test =train_test_split(x,y, test_size=0.25,random_state=150)

# print(x_train)
# print(y_train)
# print(x_test)
# print(y_test)

#2.모델구성
model=Sequential()
model.add(Dense(10,input_dim=1))
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(100))
model.add(Dense(1))

#3.컴파일, 훈련
model.compile(loss='mse',optimizer='adam')
model.fit(x_train,y_train,epochs=100, batch_size=1)

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
로스 : 22.898237228393555           test_size=0.25,random_state=50
R2 score 0.4098391955068841         10,13,80,100,30,50,1        epochs=100, batch_size=1
로스 : 23.545381546020508           test_size=0.25,random_state=50
R2 score 0.39316024652334913        10,100,10,100,10,100,1   epochs=100, batch_size=1
'''
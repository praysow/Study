#r2 0.55 ~0.6이상

from sklearn.datasets import fetch_california_housing
import time
#1.데이터
datasets = fetch_california_housing()
x =datasets.data
y =datasets.target

# print(x)   #(20640, 8)
# print(y)   #(20640,)
# print(x.shape,y.shape)
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
import numpy as np

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.7, random_state=130)
#2. 모델구성
model=Sequential()
model.add(Dense(1,input_dim=8))
model.add(Dense(100))
model.add(Dense(1))
model.add(Dense(100))
model.add(Dense(1))
model.add(Dense(100))
model.add(Dense(1))

#3.컴파일 훈련
model.compile(loss='mse',optimizer='adam')
start_time = time.time()
model.fit(x_train,y_train, epochs=600, batch_size=600, validation_split=0.3, verbose=2)
end_time = time.time()


#4.결과예측
loss=model.evaluate(x_test,y_test)
y_predict=model.predict([x_test])
result=model.predict(x)
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score
r2=r2_score(y_test,y_predict)
print("로스:",loss)
print("R2 score",r2)
print("걸린시간 :",round(end_time - start_time))

'''
로스: 0.6840656399726868
R2 score 0.4841499793360019
걸린시간 : 19



'''




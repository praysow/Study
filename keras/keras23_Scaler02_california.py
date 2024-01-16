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
#print(datasets.feature_names)  #['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
import numpy as np

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.7, random_state=130)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
scaler = MaxAbsScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

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
model.fit(x_train,y_train, epochs=1000, batch_size=600)
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
로스: 0.5083668231964111        train_size=0.9, random_state=130
R2 score 0.6050325690771254    1,100,1,100,1,100,1,epochs=5000, batch_size=600

로스: 0.5270371437072754
R2 score 0.6025642705429528         standrad

로스: 0.530608594417572
R2 score 0.5998708789067988         minmax

로스: 0.5291006565093994            robu
R2 score 0.6010080783272427

로스: 0.5959298610687256            maxabs
R2 score 0.5506126492080742
'''


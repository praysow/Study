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
#2. 모델구성
from sklearn.svm import LinearSVR
model = LinearSVR(C=100)

model.fit(x_train,y_train)

# 4.결과예측
result = model.score(x_test,y_test)
print("acc :", result)
y_predict = model.predict(x_test)
from sklearn.metrics import r2_score
r2=r2_score(y_test,y_predict)
print("R2 score",r2)

'''

로스: 0.5083668231964111        train_size=0.9, random_state=130
R2 score 0.6050325690771254    1,100,1,100,1,100,1,epochs=5000, batch_size=600

로스: 0.5432149171829224        train_size=0.7, random_state=130
R2 score 0.5903646136935563   1,100,1,100,1,100,1,epochs=5000, batch_size=600
'''



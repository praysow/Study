#r2 0.55 ~0.6이상
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import Perceptron,LogisticRegression
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
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
allAlgorithms = [
    ('LogisticRegression',LinearRegression),
    ('KNeighborsClassifier', KNeighborsRegressor),
    ('DecisionTreeClassifier',DecisionTreeRegressor),
    ('RandomForestClassifier', RandomForestRegressor)
]

# 3. 모델 훈련 및 평가
for name, algorithm in allAlgorithms:
    model = algorithm()
    model.fit(x_train, y_train)
    acc = model.score(x_test, y_test)
    print(name, '의 정확도:', acc)


'''

로스: 0.5083668231964111        train_size=0.9, random_state=130
R2 score 0.6050325690771254    1,100,1,100,1,100,1,epochs=5000, batch_size=600

로스: 0.5432149171829224        train_size=0.7, random_state=130
R2 score 0.5903646136935563   1,100,1,100,1,100,1,epochs=5000, batch_size=600
acc 1: 0.6042486641813757
acc 2: 0.12835844817304742
acc 3: 0.5995081630455021
acc 4: 0.7974553477199674
R2 score 0.6042486641813757
R2 score 0.12835844817304742
R2 score 0.5995081630455021
R2 score 0.7974553477199674
'''



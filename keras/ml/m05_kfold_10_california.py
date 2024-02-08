#r2 0.55 ~0.6이상
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import Perceptron,LogisticRegression
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor,ExtraTreesRegressor
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
from sklearn.model_selection import KFold, cross_val_score

n_split = 5
kfold = KFold(n_splits=n_split,shuffle=True, random_state=123)
#2.모델
model = ExtraTreesRegressor()   #소프트벡터머신 클래스파이어
#3.훈련
scores = cross_val_score(model,x,y,cv=kfold)

print("ACC:",scores,"\n 평균:",round(np.mean(scores),4))
'''
ACC: [0.8178997  0.83286723 0.8148894  0.79782694 0.80360131] 
 평균: 0.8134
'''



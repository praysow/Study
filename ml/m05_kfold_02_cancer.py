import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import Perceptron,LogisticRegression
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor,HistGradientBoostingClassifier
#1. 데이터
datasets= load_breast_cancer()

x = datasets.data       #(569, 30)
y = datasets.target     #(569,)
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8, random_state=450)

#3.컴파일 훈련
from sklearn.model_selection import KFold, cross_val_score

n_split = 5
# kfold = KFold(n_splits=n_split,shuffle=True, random_state=123)
from sklearn.model_selection import StratifiedKFold
kfold = StratifiedKFold(n_splits=n_split,shuffle=True, random_state=123)
#2.모델
model = HistGradientBoostingClassifier()   #소프트벡터머신 클래스파이어
#3.훈련
scores = cross_val_score(model,x,y,cv=kfold)

print("ACC:",scores,"\n 평균:",round(np.mean(scores),4))
'''
ACC: [0.97368421 0.95614035 0.96491228 0.97368421 0.94690265] 
 평균: 0.9631
'''
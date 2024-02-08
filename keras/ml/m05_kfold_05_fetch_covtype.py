import numpy as np
import pandas as pd
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers import Dense,LSTM,Conv1D,Flatten
from keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import f1_score
import time
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import Perceptron,LogisticRegression
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
#1.데이터
datasets= fetch_covtype()
x= datasets.data
y= datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.86,
                                                    random_state=5,        #346
                                                    # stratify=y_ohe1            
                                                    )

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
# scaler = MinMaxScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)

from sklearn.model_selection import KFold, cross_val_score

n_split = 5
# kfold = KFold(n_splits=n_split,shuffle=True, random_state=123)
from sklearn.model_selection import StratifiedKFold
kfold = StratifiedKFold(n_splits=n_split,shuffle=True, random_state=123)
#2.모델
model =KNeighborsClassifier()   #소프트벡터머신 클래스파이어
#3.훈련
scores = cross_val_score(model,x,y,cv=kfold)

print("ACC:",scores,"\n 평균:",round(np.mean(scores),4))
'''
ACC: [0.96832268 0.96858085 0.96873548 0.96958744 0.96873548] 
 평균: 0.9688
'''
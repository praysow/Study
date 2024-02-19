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

allAlgorithms = [
    ('LogisticRegression', LogisticRegression),
    ('KNeighborsClassifier', KNeighborsClassifier),
    ('DecisionTreeClassifier', DecisionTreeClassifier),
    ('RandomForestClassifier', RandomForestClassifier)
]

# 3. 모델 훈련 및 평가
for name, algorithm in allAlgorithms:
    model = algorithm()
    model.fit(x_train, y_train)
    acc = model.score(x_test, y_test)
    print(name, '의 정확도:', acc)

'''
accuracy_score : 0.7191856605443683
로스 : 0.6707912087440491
acc : 0.7191856503486633
걸린시간 : 83

acc 1: 0.6202207961446731
acc 2: 0.9700646652405891
acc 3: 0.9422192717169482
acc 4: 0.9568857416832632
'''
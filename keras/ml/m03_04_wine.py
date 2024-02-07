import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM,Conv1D,Flatten
from keras.utils import to_categorical
from sklearn.metrics import f1_score
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import Perceptron,LogisticRegression
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
#1.데이터
datasets= load_wine()
x= datasets.data
y= datasets.target

# # 사이킷런

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8,
                                                    random_state=383,        
                                                    # stratify=y_ohe1            
                                                    )

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
accuracy_score : 1.0
로스 : 0.015490112826228142
acc : 1.0                       아래두개

acc 1: 0.9722222222222222
acc 2: 0.6944444444444444
acc 3: 0.8611111111111112
acc 4: 0.9277220423412205
'''
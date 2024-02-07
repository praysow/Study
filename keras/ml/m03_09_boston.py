from sklearn.datasets import load_boston
datasets = load_boston()
x = datasets.data
y = datasets.target

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.9,random_state=100)
#2.모델구성
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
acc 1: 0.8360934882713208
acc 2: 0.4957847289208367
acc 3: 0.5122656219030379
acc 4: 0.8502112750069373
R2 score 0.8360934882713208
R2 score 0.4957847289208367
R2 score 0.5122656219030379
R2 score 0.8502112750069373
'''


# 로스 : 14.19102668762207          (x,y, train_size=0.9,random_state=100
# R2 score 0.8206877810194941       1,100,1,100,1,100,1,100,1epochs=5000, batch_size=10
# https://dacon.io/competitions/open/235576/leaderboard
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import Perceptron,LogisticRegression
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import pandas as pd

#1. 데이터

path= "c:\_data\dacon\ddarung\\"

train_csv = pd.read_csv(path+"train.csv",index_col=0)
test_csv= pd.read_csv(path+"test.csv",index_col=0)
submission_csv= pd.read_csv(path+"submission.csv")

train_csv=train_csv.fillna(train_csv.mean())                         #test는 dropna를 하면 안되고 결측치를 변경해줘야한다
# train_csv=train_csv.fillna(0)
test_csv=test_csv.fillna(test_csv.mean())                         #test는 dropna를 하면 안되고 결측치를 변경해줘야한다
# test_csv=test_csv.fillna(0)

x= train_csv.drop(['count'],axis=1)
y= train_csv['count']

x_train,x_test,y_train,y_test=train_test_split(x,y, train_size=0.8, random_state=6)

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

submission_16 로스 : 2469.941162109375  x,y, train_size=0.8, random_state=4
                                        10,1  epochs=2100, batch_size=32
acc 1: 0.6111335907585849
acc 2: 0.3222955497852805
acc 3: 0.6435197250151092
acc 4: 0.7932460355395848
R2 score 0.6111335907585849
R2 score 0.3222955497852805
R2 score 0.6435197250151092
R2 score 0.7932460355395848
'''
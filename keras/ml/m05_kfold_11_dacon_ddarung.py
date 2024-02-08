# https://dacon.io/competitions/open/235576/leaderboard
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import Perceptron,LogisticRegression
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor,BaggingRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

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
from sklearn.model_selection import KFold, cross_val_score

n_split = 5
kfold = KFold(n_splits=n_split,shuffle=True, random_state=123)
#2.모델
model = BaggingRegressor()
#3.훈련
scores = cross_val_score(model,x_train,y_train,cv=kfold)

print("ACC:",scores,"\n 평균:",round(np.mean(scores),4))
'''
ACC: [0.69194685 0.72550205 0.78539846 0.71966337 0.78466781] 
 평균: 0.7414
'''
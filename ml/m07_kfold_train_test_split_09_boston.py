from sklearn.datasets import load_boston
datasets = load_boston()
x = datasets.data
y = datasets.target

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from sklearn.svm import LinearSVR, SVR
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.9,random_state=100)
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer, RobustScaler
scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import StratifiedKFold,cross_val_predict
n_split = 5
kfold = KFold(n_splits=n_split,shuffle=True, random_state=123)
# from sklearn.model_selection import StratifiedKFold
# kfold = StratifiedKFold(n_splits=n_split,shuffle=True, random_state=123)

#2.모델
model =LinearRegression()   #소프트벡터머신 클래스파이어
#3.훈련
scores = cross_val_score(model, x_train, y_train, cv=kfold)
print("ACC:",scores,"\n평균:",round(np.mean(scores),4))
from sklearn.metrics import accuracy_score
y_pred = cross_val_predict(model, x_test, y_test, cv=kfold)
acc = accuracy_score(y_test, y_pred)
print("acc", acc)
'''
ACC: [0.76940727 0.86506458 0.89541677 0.92320525 0.86387876] 
 평균: 0.8634
'''

# 로스 : 14.19102668762207          (x,y, train_size=0.9,random_state=100
# R2 score 0.8206877810194941       1,100,1,100,1,100,1,100,1epochs=5000, batch_size=10
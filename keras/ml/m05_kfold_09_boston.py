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
from sklearn.model_selection import KFold, cross_val_score

n_split = 5
kfold = KFold(n_splits=n_split,shuffle=True, random_state=123)
#2.모델
model = RandomForestRegressor()   #소프트벡터머신 클래스파이어
#3.훈련
scores = cross_val_score(model,x,y,cv=kfold)

print("ACC:",scores,"\n 평균:",round(np.mean(scores),4))
'''
ACC: [0.76940727 0.86506458 0.89541677 0.92320525 0.86387876] 
 평균: 0.8634
'''


# 로스 : 14.19102668762207          (x,y, train_size=0.9,random_state=100
# R2 score 0.8206877810194941       1,100,1,100,1,100,1,100,1epochs=5000, batch_size=10
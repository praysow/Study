import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import Perceptron,LogisticRegression,SGDClassifier
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.utils import all_estimators
import warnings
from sklearn.metrics import accuracy_score
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV,HalvingGridSearchCV
warnings.filterwarnings('ignore')
import time
from sklearn.datasets import load_breast_cancer
#1. 데이터
datasets= load_breast_cancer()

x = datasets.data       #(569, 30)
y = datasets.target     #(569,)
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8, random_state=450)
parameters = [
    {'n_estimators' : [100,200], 'max_depth':[6,10,12],'min_samples_leaf' : [3,10]},
    {'max_depth' : [6,8,10,12], 'min_samples_leaf':[3,5,7,10]},
    {'min_samples_leaf':[3,5,7,10],'min_samples_split':[2,3,5,10]},
    {'min_samples_split' : [2,3,5,10]},
    {'n_jobs':[-1,2,4], 'min_samples_split' : [2,3,5,10]}
] 
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import StratifiedKFold,cross_val_predict
n_split = 5
kfold = KFold(n_splits=n_split,shuffle=True, random_state=123)
# model = SVC(C=1, kernel ='linear',degree=3)
# model = GridSearchCV(RandomForestClassifier(),parameters, cv = kfold,verbose=1,refit=True,n_jobs=-1)       #n_jobs gpu아니고 cpu
model = HalvingGridSearchCV(RandomForestClassifier(),parameters, cv = kfold,verbose=1,refit=True,n_jobs=-1,random_state=6,factor=2,min_resources=100)     #데이터를 최대한 사용하고싶다면 factor와,min_resources 를 조절하자
s_t= time.time()
model.fit(x_train,y_train)
e_t= time.time()
print("최적의 매개변수",model.best_estimator_)         
# 최적의 매개변수 SVC(C=10, kernel='linear')
print("최적의 파라미터", model.best_params_)
# 최적의 파라미터 {'C': 10, 'degree': 3, 'kernel': 'linear'}
print("베스트 스코어", model.best_score_)
#베스트 스코어 0.9916666666666666
print("model 스코어", model.score(x_test,y_test))
# model 스코어 0.9333333333333333
y_pred = model.predict(x_test)
print("acc",accuracy_score(y_test,y_pred))
#acc 0.9333333333333333
y_pred_best = model.best_estimator_.predict(x_test)
                    #  최적의 매개변수 SVC(C=10, kernel='linear').predict(x_test)
print("acc",accuracy_score(y_test,y_pred_best))
print("걸린시간",round(e_t-s_t,2),"초")
# print(pd.DataFrame(model.cv_results_))  #가로세로변환 .T

'''
n_iterations: 3
n_required_iterations: 6
n_possible_iterations: 3
min_resources_: 100
max_resources_: 455
aggressive_elimination: False
factor: 2
----------
iter: 0
n_candidates: 60
n_resources: 100
Fitting 5 folds for each of 60 candidates, totalling 300 fits
----------
iter: 1
n_candidates: 30
n_resources: 200
Fitting 5 folds for each of 30 candidates, totalling 150 fits
----------
iter: 2
n_candidates: 15
n_resources: 400
Fitting 5 folds for each of 15 candidates, totalling 75 fits
최적의 매개변수 RandomForestClassifier(min_samples_split=5, n_jobs=2)
최적의 파라미터 {'min_samples_split': 5, 'n_jobs': 2}
베스트 스코어 0.9550000000000001
model 스코어 0.9912280701754386
acc 0.9912280701754386
acc 0.9912280701754386
걸린시간 4.67 초
'''
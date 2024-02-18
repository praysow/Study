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

from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer, RobustScaler
scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

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
model = HalvingGridSearchCV(RandomForestRegressor(),parameters, cv = kfold,verbose=1,refit=True,n_jobs=-1,random_state=6,factor=3.5,min_resources=50)     #데이터를 최대한 사용하고싶다면 factor와,min_resources 를 조절하자
s_t= time.time()
model.fit(x_train,y_train)
e_t= time.time()
print("최적의 매개변수",model.best_estimator_)         
print("최적의 파라미터", model.best_params_)
print("베스트 스코어", model.best_score_)
print("model 스코어", model.score(x_test,y_test))

'''
n_iterations: 3
n_required_iterations: 4
n_possible_iterations: 3
min_resources_: 50
max_resources_: 1167
aggressive_elimination: False
factor: 3.5
----------
iter: 0
n_candidates: 60
n_resources: 50
Fitting 5 folds for each of 60 candidates, totalling 300 fits
----------
iter: 1
n_candidates: 18
n_resources: 175
Fitting 5 folds for each of 18 candidates, totalling 90 fits
----------
iter: 2
n_candidates: 6
n_resources: 612
Fitting 5 folds for each of 6 candidates, totalling 30 fits
최적의 매개변수 RandomForestRegressor(min_samples_split=3, n_jobs=-1)
최적의 파라미터 {'min_samples_split': 3, 'n_jobs': -1}
베스트 스코어 0.7763878225246026
model 스코어 0.800340625765923
'''
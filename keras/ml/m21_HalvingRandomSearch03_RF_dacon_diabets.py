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
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV,HalvingGridSearchCV,HalvingRandomSearchCV
warnings.filterwarnings('ignore')
import time
#1. 데이터

path= "c:\_data\dacon\cancer\\"
train_csv = pd.read_csv(path+"train.csv",index_col=0)
test_csv = pd.read_csv(path+"test.csv",index_col=0)
sampleSubmission_csv = pd.read_csv(path+"sample_Submission.csv")

# print("train",train_csv.shape)      #(652,9)
# print("test",test_csv.shape)       #(116, 8)
# print("sub",sampleSubmission_csv.shape) #(116,2)]

x= train_csv.drop(['Outcome'], axis=1)
y= train_csv['Outcome']

x_train,x_test,y_train,y_test=train_test_split(x,y, train_size=0.9, random_state=8)
from sklearn.metrics import accuracy_score
from sklearn.utils import all_estimators
import warnings
warnings.filterwarnings('ignore')
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
    {'n_jobs':[-1], 'min_samples_split' : [2,3,5,10]}
] 
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import StratifiedKFold,cross_val_predict
n_split = 5
kfold = KFold(n_splits=n_split,shuffle=True, random_state=123)
model = HalvingRandomSearchCV(RandomForestClassifier(),parameters, cv = kfold,verbose=1,refit=True,n_jobs=-1,random_state=6,factor=2,min_resources=10)     #데이터를 최대한 사용하고싶다면 factor와,min_resources 를 조절하자
# model = HalvingGridSearchCV(RandomForestClassifier(),parameters, cv = kfold,verbose=1,refit=True,n_jobs=-1,random_state=6,factor=2.5,min_resources=50)     #데이터를 최대한 사용하고싶다면 factor와,min_resources 를 조절하자
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
n_iterations: 6
n_required_iterations: 6
n_possible_iterations: 6
min_resources_: 10
max_resources_: 586
aggressive_elimination: False
factor: 2
----------
iter: 0
n_candidates: 52
n_resources: 10
Fitting 5 folds for each of 52 candidates, totalling 260 fits
----------
iter: 1
n_candidates: 26
n_resources: 20
Fitting 5 folds for each of 26 candidates, totalling 130 fits
----------
iter: 2
n_candidates: 13
n_resources: 40
Fitting 5 folds for each of 13 candidates, totalling 65 fits
----------
iter: 3
n_candidates: 7
n_resources: 80
Fitting 5 folds for each of 7 candidates, totalling 35 fits
----------
iter: 4
n_candidates: 4
n_resources: 160
Fitting 5 folds for each of 4 candidates, totalling 20 fits
----------
iter: 5
n_candidates: 2
n_resources: 320
Fitting 5 folds for each of 2 candidates, totalling 10 fits
최적의 매개변수 RandomForestClassifier(min_samples_split=5, n_jobs=-1)
최적의 파라미터 {'n_jobs': -1, 'min_samples_split': 5}
베스트 스코어 0.7563492063492063
model 스코어 0.7878787878787878
acc 0.7878787878787878
acc 0.7878787878787878
걸린시간 4.98 초
'''
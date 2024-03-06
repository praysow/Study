import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import Perceptron,LogisticRegression,SGDClassifier
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
import warnings
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV,HalvingGridSearchCV,HalvingRandomSearchCV
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
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split,KFold, StratifiedKFold,GridSearchCV,RandomizedSearchCV,HalvingRandomSearchCV,HalvingGridSearchCV
from xgboost import XGBRegressor
scaler=MinMaxScaler()
scaler=StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits,shuffle=True,random_state=123)


parameters = {
    'n_estimators' : [100,200,300] ,
    'learning_rate' : [0.1,0.2,0.3],
    'max_depth' : [None,2,3,4,5],
    'gamma' : [0,1,2,3,4],
    'min_child_weight' : [0,0.001,0.01,0.1,0.5,1,5,10,100],
    'subsample' : [0,0.1,0.2,0.3,0.5,1],
    'colsample_bytree' : [0,0.1,0.2,0.3,1],
    'colsample_bylevel' : [0,0.1,0.2,0.3,1],
    'colsample_bynode' : [0,0.1,0.2,0.3,1],
    'reg_alpha' : [0,0.1,0.01,0.001,1,2,10],
    'reg_lambda' : [0,0.1,0.01,0.001,1,2,10],
    'random_state' : [123]
    }
#2. 모델
xgb = XGBRegressor(random_state=123)
model = RandomizedSearchCV(xgb,parameters,cv=kfold,n_jobs=22)

#3. 훈련
model.fit(x_train,y_train)

#4. 평가,예측
print("최상의 매개변수 :", model.best_estimator_)
print("최상의 매개변수 :", model.best_params_)
print("최상의 점수 :", model.best_score_)

results = model.score(x_test,y_test)
y_pred = model.predict(x_test)
print("result",results)
'''
최상의 매개변수 : XGBRegressor(base_score=None, booster=None, callbacks=None, colsample_bylevel=0,
             colsample_bynode=1, colsample_bytree=1, device=None,
             early_stopping_rounds=None, enable_categorical=False,
             eval_metric=None, feature_types=None, gamma=2, grow_policy=None,
             importance_type=None, interaction_constraints=None,
             learning_rate=0.3, max_bin=None, max_cat_threshold=None,
             max_cat_to_onehot=None, max_delta_step=None, max_depth=3,
             max_leaves=None, min_child_weight=10, missing=nan,
             monotone_constraints=None, multi_strategy=None, n_estimators=100,
             n_jobs=None, num_parallel_tree=None, random_state=123, ...)
최상의 매개변수 : {'subsample': 1, 'reg_lambda': 1, 'reg_alpha': 2, 'random_state': 123, 'n_estimators': 100, 'min_child_weight': 10, 'max_depth': 3, 'learning_rate': 0.3, 'gamma': 2, 'colsample_bytree': 1, 'colsample_bynode': 1, 'colsample_bylevel': 0}
최상의 점수 : 0.708760770972444
result 0.7504308992605642
'''
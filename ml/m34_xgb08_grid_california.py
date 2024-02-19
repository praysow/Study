import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import warnings
from sklearn.metrics import accuracy_score
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV,HalvingGridSearchCV,HalvingRandomSearchCV
warnings.filterwarnings('ignore')
import time
from sklearn.datasets import fetch_california_housing

#1.데이터
datasets = fetch_california_housing()
x =datasets.data
y =datasets.target

# print(x)   #(20640, 8)
# print(y)   #(20640,)
# print(x.shape,y.shape)
#print(datasets.feature_names)  #['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
import numpy as np

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.7, random_state=130)
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
kfold = KFold(n_splits=n_splits,shuffle=True,random_state=123)


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
최상의 매개변수 : XGBRegressor(base_score=None, booster=None, callbacks=None, colsample_bylevel=1,
             colsample_bynode=1, colsample_bytree=1, device=None,
             early_stopping_rounds=None, enable_categorical=False,
             eval_metric=None, feature_types=None, gamma=4, grow_policy=None,
             importance_type=None, interaction_constraints=None,
             learning_rate=0.3, max_bin=None, max_cat_threshold=None,
             max_cat_to_onehot=None, max_delta_step=None, max_depth=5,
             max_leaves=None, min_child_weight=5, missing=nan,
             monotone_constraints=None, multi_strategy=None, n_estimators=100,
             n_jobs=None, num_parallel_tree=None, random_state=123, ...)
최상의 매개변수 : {'subsample': 0.5, 'reg_lambda': 1, 'reg_alpha': 0, 'random_state': 123, 'n_estimators': 100, 'min_child_weight': 5, 'max_depth': 5, 'learning_rate': 0.3, 'gamma': 4, 'colsample_bytree': 1, 'colsample_bynode': 1, 'colsample_bylevel': 1}
최상의 점수 : 0.7957955395732581
result 0.7869970845316947
'''
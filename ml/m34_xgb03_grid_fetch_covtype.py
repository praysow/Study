import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import warnings
from sklearn.metrics import accuracy_score
warnings.filterwarnings('ignore')
import time
from sklearn.datasets import fetch_covtype

#1.데이터
datasets= fetch_covtype()
x= datasets.data
y= datasets.target

y -= 1

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.86,
                                                    random_state=5,        #346
                                                    # stratify=y_ohe1            
                                                    )
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split,KFold, StratifiedKFold,GridSearchCV,RandomizedSearchCV,HalvingRandomSearchCV,HalvingGridSearchCV
from xgboost import XGBClassifier
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
xgb = XGBClassifier(random_state=123)
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
최상의 매개변수 : XGBClassifier(base_score=None, booster=None, callbacks=None,
              colsample_bylevel=1, colsample_bynode=1, colsample_bytree=1,
              device=None, early_stopping_rounds=None, enable_categorical=False,
              eval_metric=None, feature_types=None, gamma=1, grow_policy=None,
              importance_type=None, interaction_constraints=None,
              learning_rate=0.1, max_bin=None, max_cat_threshold=None,
              max_cat_to_onehot=None, max_delta_step=None, max_depth=5,
              max_leaves=None, min_child_weight=0, missing=nan,
              monotone_constraints=None, multi_strategy=None, n_estimators=100,
              n_jobs=None, num_parallel_tree=None, objective='multi:softprob', ...)
최상의 매개변수 : {'subsample': 1, 'reg_lambda': 1, 'reg_alpha': 0, 'random_state': 123, 'n_estimators': 100, 'min_child_weight': 0, 'max_depth': 5, 'learning_rate': 0.1, 'gamma': 1, 'colsample_bytree': 1, 'colsample_bynode': 1, 'colsample_bylevel': 1}
최상의 점수 : 0.7871275041527408
result 0.7888298787834084
'''
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import r2_score

# 데이터 로드
path = "c:\_data\dacon\ddarung\\"
train_csv = pd.read_csv(path+"train.csv", index_col=0)
test_csv = pd.read_csv(path+"test.csv", index_col=0)
submission_csv = pd.read_csv(path+"submission.csv")

# 결측치 처리
train_csv = train_csv.fillna(train_csv.mean())
test_csv = test_csv.fillna(test_csv.mean())

# 특성과 타겟 분리
x = train_csv.drop(['count'], axis=1)
y = train_csv['count']
x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.9,random_state=1)
from bayes_opt import BayesianOptimization
import time
bayesian_params = {
    'learning_rate' : (0.001,1),
    'max_depth' : (3,10),
    'num_leaves' : (24,40),
    'min_child_samples' : (10,200),
    'min_child_weight': (1,50),
    'subsample' : (0.5,1),
    'colsample_bytree' : (0.5,1),
    'max_bin' : (9,500),
    'reg_lambda' : (-0.001,10),
    'reg_alpha': (0.01,50)
}

def xgb_hamsu(learning_rate,max_depth,num_leaves,min_child_samples,min_child_weight,subsample,colsample_bytree,max_bin,reg_lambda,reg_alpha):
    params = {
    'n_estimators' : 100,
    'learning_rate' : learning_rate,
    'max_depth' : int(round(max_depth)),
    'num_leaves' : int(round(num_leaves)),
    'min_child_samples' : int(round(min_child_samples)),
    'min_child_weight': int(round(min_child_weight)),
    'subsample' : max(min(subsample,1),0),
    'colsample_bytree' : colsample_bytree,
    'max_bin' : max(int(round(max_bin)),10),
    'reg_lambda' : max(reg_lambda,0),
    'reg_alpha': reg_alpha
    }
    model = XGBRegressor(**params,n_jobs = -1)
    model.fit(x_train,y_train,
              eval_set=[(x_train,y_train),(x_test,y_test)],
            #   eval_metric='mlogloss',
              verbose = 0,
              early_stopping_rounds= 50
              )
    y_pred = model.predict(x_test)
    result = r2_score(y_test,y_pred)
    return result
s_t = time.time()
bay = BayesianOptimization(
    f= xgb_hamsu,
    pbounds=bayesian_params,
    random_state=777   
    )
n_iter = 100
bay.maximize(init_points=5,n_iter=n_iter)
e_t = time.time()
print(bay.max)
print("시간:",round((e_t-s_t),2),"초")
'''
{'target': 0.8255955327830022, 'params': {'colsample_bytree': 0.8905076005077042, 'learning_rate': 0.2645482198571758, 'max_bin': 333.0849969535737, 'max_depth': 6.130702495369259, 'min_child_samples': 26.04473909687385, 'min_child_weight': 2.0010848957158265, 'num_leaves': 35.722500241617894, 'reg_alpha': 9.987602411510274, 'reg_lambda': 7.758760975034975, 'subsample': 0.9034516698680519}}
시간: 14.81 초
'''
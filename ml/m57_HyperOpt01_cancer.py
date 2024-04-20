import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler,PolynomialFeatures
from xgboost import XGBClassifier
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from bayes_opt import BayesianOptimization
from hyperopt import hp, fmin , tpe, Trials, STATUS_OK

import time
#1.데이터
s_t = time.time()
x,y = load_breast_cancer(return_X_y=True)
x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.9,random_state=1)
search_space = {
    'learning_rate' : hp.quniform('learning_rate',0.001,1),
    'max_depth' : hp.quniform('max_depth',3,10),
    'num_leaves' : hp.quniform('num_leaves',24,40),
    'min_child_samples' : hp.quniform('min_child_samples',10,200),
    'min_child_weight': hp.quniform('min_child_weight',1,50),
    'subsample' : hp.quniform('subsample',0.5,1),
    'colsample_bytree' : hp.quniform('colsample_bytree',0.5,1),
    'max_bin' : hp.quniform('max_bin',9,500),
    'reg_lambda' : hp.quniform('reg_lambda',-0.001,10),
    'reg_alpha': hp.quniform('reg_alpha',0.01,50)
}

def xgb_hamsu(learning_rate,max_depth,num_leaves,min_child_samples,min_child_weight,subsample,colsample_bytree,max_bin,reg_lambda,reg_alpha):
    params = {
    'n_estimators' : 100,
    'learning_rate' : search_space['learning_rate'],
    'max_depth' : int(search_space['max_depth']),
    'num_leaves' : int(search_space['num_leaves']),
    'min_child_samples' : int(search_space['min_child_samples']),
    'min_child_weight': int(search_space['min_child_weight']),
    'subsample' : max(min(search_space['subsample'],1),0),
    'colsample_bytree' : search_space['colsample_bytree,'],
    'max_bin' : max(int(search_space['max_bin'],10)),
    'reg_lambda' : max(search_space['reg_lambda'],0),
    'reg_alpha': search_space['reg_alpha']
    }
    trial_val = Trials()


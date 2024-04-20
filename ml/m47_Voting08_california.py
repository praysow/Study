import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import warnings
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.datasets import fetch_california_housing

# 데이터 로드
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target
x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.9,random_state=1)
from sklearn.preprocessing import MinMaxScaler
sclaer = MinMaxScaler().fit(x_train)
x_train = sclaer.transform(x_train)
x_test = sclaer.transform(x_test)
from xgboost import XGBRegressor
# model 
from sklearn.ensemble import VotingRegressor, RandomForestRegressor
model = VotingRegressor([
    # ('LR',LogisticRegression()),
    ('RF',RandomForestRegressor()),
    ('XGB',XGBRegressor()),
    ])

# fit & pred
model.fit(x_train,y_train,)

result = model.score(x_test,y_test)
print("Score: ",result)

pred = model.predict(x_test)
acc = r2_score(y_test,pred)
print("ACC: ",acc)
'''
Score:  0.8385207559525152
ACC:  0.8385207559525152
'''
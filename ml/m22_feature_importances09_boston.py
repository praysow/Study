from sklearn.datasets import load_boston
datasets = load_boston()
x = datasets.data
y = datasets.target

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import all_estimators
import warnings
from sklearn.metrics import accuracy_score
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV,HalvingGridSearchCV,HalvingRandomSearchCV
warnings.filterwarnings('ignore')
import time
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.9,random_state=100)

from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer, RobustScaler
scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

from xgboost import XGBRegressor
model3 = XGBRegressor(random_state = 100)
model3.fit(x_train,y_train)

# 4.결과예측
result3 = model3.score(x_test,y_test)
y_predict3 = model3.predict(x_test)

print("acc :", result3)

print(model3.feature_importances_)

'''
acc : 0.8698687514229004
[0.02209632 0.00217058 0.01532184 0.00275956 0.04145647 0.29855964
 0.01512801 0.09086201 0.01302536 0.0340099  0.02333337 0.00796757
 0.43330932]

'''
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
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
from xgboost import XGBClassifier
model3 = XGBClassifier(random_state = 100)
model3.fit(x_train,y_train)

# 4.결과예측
result3 = model3.score(x_test,y_test)
y_predict3 = model3.predict(x_test)

print("acc :", result3)

print(model3.feature_importances_)
'''
acc : 0.9736842105263158
[0.001178   0.02108569 0.         0.         0.01457767 0.03457553
 0.03944778 0.21718071 0.00224723 0.00693146 0.01092922 0.01530164
 0.02537417 0.00840552 0.00736203 0.00540313 0.01006659 0.00055569
 0.00167762 0.01573942 0.10944817 0.01264871 0.06917711 0.03029872
 0.00367436 0.00434861 0.0268101  0.2942366  0.00150074 0.00981779]
'''
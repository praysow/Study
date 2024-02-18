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
from sklearn.datasets import load_wine

#1.데이터
datasets= load_wine()
x= datasets.data
y= datasets.target

# # 사이킷런

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8,
                                                    random_state=383,        
                                                    # stratify=y_ohe1            
                                                    )
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer, RobustScaler
scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

from xgboost import XGBClassifier
model3 = XGBClassifier(random_state = 100)
model3.fit(x_train,y_train)

# 4.결과예측
result3 = model3.score(x_test,y_test)
y_predict3 = model3.predict(x_test)

print("acc :", result3)

print(model3.feature_importances_)

'''
acc : 0.9444444444444444
[0.09167349 0.0517154  0.01133628 0.00607679 0.05680155 0.00576556
 0.13963431 0.00860458 0.02458011 0.15605102 0.02968677 0.2573468
 0.1607274 ]
'''
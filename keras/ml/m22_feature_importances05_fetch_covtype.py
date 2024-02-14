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
from sklearn.datasets import fetch_covtype

#1.데이터
datasets= fetch_covtype()
x= datasets.data
y= datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.86,
                                                    random_state=5,        #346
                                                    stratify=y            
                                                    )

from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer, RobustScaler
scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# 클래스 레이블을 0부터 시작하여 순차적으로 증가하는 값으로 변환
y_train -= 1
y_test -= 1

from xgboost import XGBClassifier
model3 = XGBClassifier(random_state = 100)
model3.fit(x_train,y_train)

# 4.결과예측
result3 = model3.score(x_test,y_test)
y_predict3 = model3.predict(x_test)

print("acc :", result3)

print(model3.feature_importances_)

'''
acc : 0.871001450665093
[0.0943685  0.00709773 0.00428762 0.01368964 0.00746966 0.01318873
 0.00829285 0.01152921 0.00566924 0.01183092 0.06055282 0.02626602
 0.0311599  0.02002065 0.0041614  0.04252253 0.021832   0.03759216
 0.00554146 0.00532906 0.00159734 0.01424221 0.00665519 0.01128453
 0.01059862 0.04689567 0.01204702 0.00420299 0.         0.00602594
 0.00850996 0.00627456 0.00655324 0.01605931 0.02152055 0.0541005
 0.02864109 0.01622993 0.01183418 0.00635261 0.01791127 0.00268712
 0.02802787 0.01969909 0.02329826 0.04127464 0.01771121 0.00596195
 0.01904675 0.00290247 0.00945527 0.03637871 0.04048762 0.01313032]
'''
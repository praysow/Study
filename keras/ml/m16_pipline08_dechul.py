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
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV,HalvingGridSearchCV
warnings.filterwarnings('ignore')
import time
from sklearn.preprocessing import LabelEncoder
#1.데이터
path= "c:\_data\dacon\dechul\\"
train_csv=pd.read_csv(path+"train.csv",index_col=0)
test_csv=pd.read_csv(path+"test.csv",index_col=0)
sample_csv=pd.read_csv(path+"sample_submission.csv")
x= train_csv.drop(['대출등급'],axis=1)
y= train_csv['대출등급']


lb=LabelEncoder()
lb.fit(x['대출기간'])
x['대출기간'] = lb.transform(x['대출기간'])
lb.fit(x['근로기간'])
x['근로기간'] = lb.transform(x['근로기간'])
lb.fit(x['주택소유상태'])
x['주택소유상태'] = lb.transform(x['주택소유상태'])
lb.fit(x['대출목적'])
x['대출목적'] = lb.transform(x['대출목적'])

lb.fit(test_csv['대출기간'])
test_csv['대출기간'] =lb.transform(test_csv['대출기간'])

lb.fit(test_csv['근로기간'])
test_csv['근로기간'] =lb.transform(test_csv['근로기간'])

lb.fit(test_csv['주택소유상태'])
test_csv['주택소유상태'] =lb.transform(test_csv['주택소유상태'])

lb.fit(test_csv['대출목적'])
test_csv['대출목적'] =lb.transform(test_csv['대출목적'])


x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.85,random_state=100 ,
                                              #  stratify=y
                                               )
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer, RobustScaler
from sklearn.pipeline import make_pipeline
models = [
    (HalvingGridSearchCV, {"estimator": RandomForestClassifier(), "param_grid": {"n_estimators": [5], "n_jobs": [2]}, "cv": 5}),
    (GridSearchCV, {"estimator": RandomForestClassifier(), "param_grid": {"n_estimators": [5], "n_jobs": [2]}, "cv": 5}),
    (RandomizedSearchCV, {"estimator": RandomForestClassifier(), "param_distributions": {"n_estimators": [5], "n_jobs": [2]}, "cv": 5})
]

# 모델 생성 및 학습
for i, (search_cv, params) in enumerate(models, 1):
    model = make_pipeline(MinMaxScaler(), search_cv(**params))
    model.fit(x_train, y_train)
    result = model.score(x_test, y_test)
    print(f"Model {i} - Accuracy: {result}")
'''
Model 1 - Accuracy: 0.7151263412945656
Model 2 - Accuracy: 0.7289719626168224
Model 3 - Accuracy: 0.7177570093457943
'''
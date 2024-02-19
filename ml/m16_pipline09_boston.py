from sklearn.datasets import load_boston
datasets = load_boston()
x = datasets.data
y = datasets.target

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
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.9,random_state=100)
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer, RobustScaler
from sklearn.pipeline import make_pipeline
models = [
    (HalvingGridSearchCV, {"estimator": RandomForestRegressor(), "param_grid": {"n_estimators": [5], "n_jobs": [2]}, "cv": 5}),
    (GridSearchCV, {"estimator": RandomForestRegressor(), "param_grid": {"n_estimators": [5], "n_jobs": [2]}, "cv": 5}),
    (RandomizedSearchCV, {"estimator": RandomForestRegressor(), "param_distributions": {"n_estimators": [5], "n_jobs": [2]}, "cv": 5})
]

# 모델 생성 및 학습
for i, (search_cv, params) in enumerate(models, 1):
    model = make_pipeline(MinMaxScaler(), search_cv(**params))
    model.fit(x_train, y_train)
    result = model.score(x_test, y_test)
    print(f"Model {i} - Accuracy: {result}")
'''
Model 1 - Accuracy: 0.7493135480119981
Model 2 - Accuracy: 0.8803383369230566
Model 3 - Accuracy: 0.8982067680598317
'''

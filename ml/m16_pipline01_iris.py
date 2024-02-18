import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler, MaxAbsScaler
from sklearn.pipeline import make_pipeline
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV, GridSearchCV, RandomizedSearchCV

# 1. 데이터
x, y = load_iris(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8,
                                                    random_state=1,
                                                    stratify=y)

# 모델 생성 및 하이퍼파라미터 탐색 방법 설정
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
Model 1 - Accuracy: 0.958
Model 2 - Accuracy: 0.98
Model 3 - Accuracy: 0.962
'''
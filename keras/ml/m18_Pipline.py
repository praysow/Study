#데이터셋마다 최상의 파라미터를 알고있다
#


import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.preprocessing import RobustScaler,MinMaxScaler,StandardScaler,MaxAbsScaler
from sklearn.pipeline import make_pipeline,Pipeline
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV,HalvingGridSearchCV

# 1.데이터
x,y = load_iris(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8,
                                                    random_state=1,         #850:acc=1
                                                    stratify=y              #stratify는 분류에서만 사용
                                                    )
# scaler = RobustScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)
model = make_pipeline(MinMaxScaler(),HalvingGridSearchCV(RandomForestRegressor(),param_grid={"n_estimators": [5],"n_jobs": [2]},cv=5))
# model = Pipeline([('MinMax',MinMaxScaler()),('RF',RandomForestClassifier())])
model.fit(x_train,y_train)

# 4.결과예측
result = model.score(x_test,y_test)
print("acc :", result)
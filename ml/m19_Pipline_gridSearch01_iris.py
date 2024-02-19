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
parameters = [
    {'RF__n_estimators' : [100,200], 'RF__max_depth':[6,10,12],'RF__min_samples_leaf' : [3,10]},
    {'RF__max_depth' : [6,8,10,12], 'RF__min_samples_leaf':[3,5,7,10]},
    {'RF__min_samples_leaf':[3,5,7,10],'RF__min_samples_split':[2,3,5,10]},
    {'RF__min_samples_split' : [2,3,5,10]},
    {'RF__n_jobs':[-1,2,4], 'RF__min_samples_split' : [2,3,5,10]}
] 
pipe = Pipeline([('MinMax',MinMaxScaler()),('RF',RandomForestClassifier())])
model1 = GridSearchCV(pipe,parameters,cv=5,verbose=1)
model2 = RandomizedSearchCV(pipe,parameters,cv=5,verbose=1)
model3 = HalvingGridSearchCV(pipe,parameters,cv=5,verbose=1,factor=3.5,min_resources=10)
model = model3
model.fit(x_train,y_train)

# 4.결과예측
result = model.score(x_test,y_test)
print("acc :", result)

'''
n_iterations: 2
n_required_iterations: 4
n_possible_iterations: 2
min_resources_: 10
max_resources_: 120
aggressive_elimination: False
factor: 3.5
----------
iter: 0
n_candidates: 60
n_resources: 10
Fitting 5 folds for each of 60 candidates, totalling 300 fits
----------
iter: 1
n_candidates: 18
n_resources: 35
Fitting 5 folds for each of 18 candidates, totalling 90 fits
acc : 0.9666666666666667
'''
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
from sklearn.datasets import fetch_covtype

#1.데이터
datasets= fetch_covtype()
x= datasets.data
y= datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.86,
                                                    random_state=5,        #346
                                                    # stratify=y_ohe1            
                                                    )
parameters = [
    {'RF__n_estimators' : [100,200], 'RF__max_depth':[6,10,12],'RF__min_samples_leaf' : [3,10]},
    {'RF__max_depth' : [6,8,10,12], 'RF__min_samples_leaf':[3,5,7,10]},
    {'RF__min_samples_leaf':[3,5,7,10],'RF__min_samples_split':[2,3,5,10]},
    {'RF__min_samples_split' : [2,3,5,10]},
    {'RF__n_jobs':[-1,2,4], 'RF__min_samples_split' : [2,3,5,10]}
] 
from sklearn.preprocessing import RobustScaler,MinMaxScaler,StandardScaler,MaxAbsScaler
from sklearn.pipeline import make_pipeline,Pipeline
pipe = Pipeline([('MinMax',MinMaxScaler()),('RF',RandomForestClassifier())])
model1 = GridSearchCV(pipe,parameters,cv=5,verbose=1)
model2 = RandomizedSearchCV(pipe,parameters,cv=5,verbose=1)
model3 = HalvingGridSearchCV(pipe,parameters,cv=5,verbose=1)
model = model3
model.fit(x_train,y_train)

# 4.결과예측
result = model.score(x_test,y_test)
print("acc :", result)
'''
n_iterations: 4
n_required_iterations: 4
n_possible_iterations: 4
min_resources_: 18506
max_resources_: 499670
aggressive_elimination: False
factor: 3
----------
iter: 0
n_candidates: 60
n_resources: 18506
Fitting 5 folds for each of 60 candidates, totalling 300 fits
----------
iter: 1
n_candidates: 20
n_resources: 55518
Fitting 5 folds for each of 20 candidates, totalling 100 fits
----------
iter: 2
n_candidates: 7
n_resources: 166554
Fitting 5 folds for each of 7 candidates, totalling 35 fits
----------
iter: 3
n_candidates: 3
n_resources: 499662
Fitting 5 folds for each of 3 candidates, totalling 15 fits
acc : 0.9565292222959848
'''
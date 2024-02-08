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
from sklearn.model_selection import GridSearchCV
warnings.filterwarnings('ignore')
import time

# 1.데이터
x,y = load_iris(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8,
                                                    random_state=450,         #850:acc=1
                                                    stratify=y              #stratify는 분류에서만 사용
                                                    )
parameters = [
    {"C":[1,10,100,1000],"kernel":["linear"],"degree":[3,4,5]},
    {"C":[1,10,100],"kernel":["rbf"],"gamma":[3,4,5]},
    {"C":[1,10,100,1000],"kernel":["sigmoid"],
      "gamma":[0.01,0.001,0.0001]}                              #총 32번 훈련
]   
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import StratifiedKFold,cross_val_predict
n_split = 5
kfold = KFold(n_splits=n_split,shuffle=True, random_state=123)
# model = SVC(C=1, kernel ='linear',degree=3)
model = GridSearchCV(SVC(),parameters, cv = kfold,verbose=1,refit=True,n_jobs=-1)       #n_jobs gpu아니고 cpu
s_t= time.time()
model.fit(x_train,y_train)
e_t= time.time()
print("최적의 매개변수",model.best_estimator_)         
# 최적의 매개변수 SVC(C=10, kernel='linear')
print("최적의 파라미터", model.best_params_)
# 최적의 파라미터 {'C': 10, 'degree': 3, 'kernel': 'linear'}
print("베스트 스코어", model.best_score_)
#베스트 스코어 0.9916666666666666
print("model 스코어", model.score(x_test,y_test))
# model 스코어 0.9333333333333333
y_pred = model.predict(x_test)
print("acc",accuracy_score(y_test,y_pred))
#acc 0.9333333333333333
y_pred_best = model.best_estimator_.predict(x_test)
                    #  최적의 매개변수 SVC(C=10, kernel='linear').predict(x_test)
print("acc",accuracy_score(y_test,y_pred_best))
print("걸린시간",round(e_t-s_t,2),"초")
print(pd.DataFrame(model.cv_results_))  #가로세로변환 .T





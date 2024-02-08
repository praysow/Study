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
warnings.filterwarnings('ignore')
# 1.데이터
x,y = load_iris(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8,
                                                    random_state=450,         #850:acc=1
                                                    stratify=y              #stratify는 분류에서만 사용
                                                    )
best_score = 0
for gamma in [0.001,0.01,0.1,1,10,100]:             #리스트는 이터레이터형태
    for C in [0.001,0.01,0.1,1,10,100]:
        model = SVC(gamma=gamma,C=C)
        model.fit(x_train,y_train)
        
        score = model.score(x_test,y_test)
        
        if score > best_score:
            best_score = score
            best_parameters = {'C':C,'gamma':gamma}

print("최고점수,{:.2f}".format(best_score))
print("최적 매개변수(파리미터) :",best_parameters)

#독립변수x 종속변수y



'''
CC: [0.86666667 1.         0.83333333 0.56666667 0.63333333] 
 평균: 0.78
'''
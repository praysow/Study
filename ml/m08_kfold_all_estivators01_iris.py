import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import Perceptron,LogisticRegression,SGDClassifier
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
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer, RobustScaler
scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import StratifiedKFold,cross_val_predict

allAlgorithms = all_estimators(type_filter='classifier')
# allAlgorithms = all_estimators(type_filter='regressor')


n_split = 5
kfold = KFold(n_splits=n_split,shuffle=True, random_state=123)
#2.모델훈련
for name, algorithm in allAlgorithms:
    try:
        #2.모델
        model = algorithm()
        #3.훈련
       
        scores = cross_val_score(model, x_train, y_train, cv=kfold)
        print("ACC:",scores,"\n평균:",round(np.mean(scores),4))

        y_pred = cross_val_predict(model, x_test, y_test, cv=kfold)
        acc = accuracy_score(y_test, y_pred)
        print("acc", acc)
    except:
        print(name, '실패')
        continue



'''
CC: [0.86666667 1.         0.83333333 0.56666667 0.63333333] 
 평균: 0.78
'''
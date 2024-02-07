import numpy as np
import pandas as pd
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers import Dense,LSTM,Conv1D,Flatten
from keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import f1_score
import time
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import Perceptron,LogisticRegression
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
#1.데이터
datasets= fetch_covtype()
x= datasets.data
y= datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.86,
                                                    random_state=5,        #346
                                                    # stratify=y_ohe1            
                                                    )

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
# scaler = MinMaxScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)

from sklearn.utils import all_estimators
import warnings
warnings.filterwarnings('ignore')
#2.모델구성
allAlgorithms = all_estimators(type_filter='classifier')
# allAlgorithms = all_estimators(type_filter='regressor')

#3.모델훈련
# print(allAlgorithms)
# print(len(allAlgorithms))   #41개
for name, algorithm in allAlgorithms:
    try:
        #2.모델
        model = algorithm()
        #3.훈련
        model.fit(x_train,y_train)

        acc = model.score(x_test,y_test)
        print(name,'의 정답률:',acc)
    except:
        print(name, '실패')
        continue
'''
AdaBoostClassifier 의 정답률: 0.44959553490201865
BaggingClassifier 의 정답률: 0.9629711588109464
BernoulliNB 의 정답률: 0.6333751321580487
CalibratedClassifierCV 의 정답률: 0.7073590519043053
CategoricalNB 실패
ClassifierChain 실패
ComplementNB 실패
DecisionTreeClassifier 의 정답률: 0.9429446042634801
DummyClassifier 의 정답률: 0.4877922844287084
ExtraTreeClassifier 의 정답률: 0.8788202896412677
ExtraTreesClassifier 의 정답률: 0.9545868063239163
GaussianNB 의 정답률: 0.45788153721324776
GaussianProcessClassifier 실패
GradientBoostingClassifier 의 정답률: 0.774556809520297
HistGradientBoostingClassifier 의 정답률: 0.8365174202748887
KNeighborsClassifier 의 정답률: 0.9700646652405891
LabelPropagation 실패
LabelSpreading 실패
LinearDiscriminantAnalysis 의 정답률: 0.6788129133780827
'''
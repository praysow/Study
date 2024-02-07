import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM,Conv1D,Flatten
from keras.utils import to_categorical
from sklearn.metrics import f1_score
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import Perceptron,LogisticRegression
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
#1.데이터
datasets= load_wine()
x= datasets.data
y= datasets.target

# # 사이킷런

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8,
                                                    random_state=383,        
                                                    # stratify=y_ohe1            
                                                    )

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
AdaBoostClassifier 의 정답률: 0.8055555555555556
BaggingClassifier 의 정답률: 0.9166666666666666
BernoulliNB 의 정답률: 0.3611111111111111
CalibratedClassifierCV 의 정답률: 0.8888888888888888
CategoricalNB 실패
ClassifierChain 실패
ComplementNB 의 정답률: 0.5
DecisionTreeClassifier 의 정답률: 0.8611111111111112
DummyClassifier 의 정답률: 0.3611111111111111
ExtraTreeClassifier 의 정답률: 0.8055555555555556
ExtraTreesClassifier 의 정답률: 0.9722222222222222
GaussianNB 의 정답률: 0.9722222222222222
GaussianProcessClassifier 의 정답률: 0.5833333333333334
GradientBoostingClassifier 의 정답률: 0.8888888888888888
HistGradientBoostingClassifier 의 정답률: 0.9444444444444444
KNeighborsClassifier 의 정답률: 0.6944444444444444
LabelPropagation 의 정답률: 0.4444444444444444
LabelSpreading 의 정답률: 0.4444444444444444
LinearDiscriminantAnalysis 의 정답률: 0.9722222222222222
LinearSVC 의 정답률: 0.9166666666666666
LogisticRegression 의 정답률: 0.9722222222222222
LogisticRegressionCV 의 정답률: 0.9722222222222222
MLPClassifier 의 정답률: 0.3888888888888889
MultiOutputClassifier 실패
MultinomialNB 의 정답률: 0.8055555555555556
NearestCentroid 의 정답률: 0.6666666666666666
NuSVC 의 정답률: 0.9444444444444444
OneVsOneClassifier 실패
OneVsRestClassifier 실패
OutputCodeClassifier 실패
PassiveAggressiveClassifier 의 정답률: 0.4722222222222222
Perceptron 의 정답률: 0.5
QuadraticDiscriminantAnalysis 의 정답률: 1.0
RadiusNeighborsClassifier 실패
RandomForestClassifier 의 정답률: 0.9444444444444444
RidgeClassifier 의 정답률: 1.0
RidgeClassifierCV 의 정답률: 1.0
SGDClassifier 의 정답률: 0.5555555555555556
SVC 의 정답률: 0.6388888888888888
StackingClassifier 실패
VotingClassifier 실패
'''
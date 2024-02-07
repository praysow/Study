import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import Perceptron,LogisticRegression
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.utils import all_estimators
import warnings
warnings.filterwarnings('ignore')
# 1.데이터
x,y = load_iris(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8,
                                                    random_state=1,         #850:acc=1
                                                    stratify=y              #stratify는 분류에서만 사용
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
AdaBoostClassifier 의 정답률: 0.9666666666666667
BaggingClassifier 의 정답률: 0.9666666666666667
BernoulliNB 의 정답률: 0.3333333333333333
CalibratedClassifierCV 의 정답률: 0.9333333333333333
CategoricalNB 의 정답률: 0.9
ClassifierChain 실패
ComplementNB 의 정답률: 0.6666666666666666
DecisionTreeClassifier 의 정답률: 0.9666666666666667
DummyClassifier 의 정답률: 0.3333333333333333
ExtraTreeClassifier 의 정답률: 0.9666666666666667
ExtraTreesClassifier 의 정답률: 0.9666666666666667
GaussianNB 의 정답률: 0.9666666666666667
GaussianProcessClassifier 의 정답률: 0.9666666666666667
GradientBoostingClassifier 의 정답률: 0.9666666666666667
HistGradientBoostingClassifier 의 정답률: 0.9666666666666667
KNeighborsClassifier 의 정답률: 0.9666666666666667
LabelPropagation 의 정답률: 0.9666666666666667
LabelSpreading 의 정답률: 0.9666666666666667
LinearDiscriminantAnalysis 의 정답률: 1.0
LinearSVC 의 정답률: 1.0
LogisticRegression 의 정답률: 0.9666666666666667
LogisticRegressionCV 의 정답률: 0.9666666666666667
MLPClassifier 의 정답률: 1.0
MultiOutputClassifier 실패
MultinomialNB 의 정답률: 1.0
NearestCentroid 의 정답률: 0.8666666666666667
NuSVC 의 정답률: 0.9333333333333333
OneVsOneClassifier 실패
OneVsRestClassifier 실패
OutputCodeClassifier 실패
PassiveAggressiveClassifier 의 정답률: 0.7
Perceptron 의 정답률: 1.0
QuadraticDiscriminantAnalysis 의 정답률: 1.0
RadiusNeighborsClassifier 의 정답률: 0.9
RandomForestClassifier 의 정답률: 0.9666666666666667
RidgeClassifier 의 정답률: 0.8666666666666667
RidgeClassifierCV 의 정답률: 0.8666666666666667
SGDClassifier 의 정답률: 1.0
SVC 의 정답률: 0.9666666666666667
StackingClassifier 실패
VotingClassifier 실패
'''
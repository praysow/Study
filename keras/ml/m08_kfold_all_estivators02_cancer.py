import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import Perceptron,LogisticRegression
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
#1. 데이터
datasets= load_breast_cancer()

x = datasets.data       #(569, 30)
y = datasets.target     #(569,)
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8, random_state=450)

#3.컴파일 훈련
from sklearn.metrics import accuracy_score
from sklearn.utils import all_estimators
import warnings
warnings.filterwarnings('ignore')
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
AdaBoostClassifier 의 정답률: 0.9736842105263158
BaggingClassifier 의 정답률: 0.956140350877193
BernoulliNB 의 정답률: 0.6052631578947368
CalibratedClassifierCV 의 정답률: 0.9649122807017544
CategoricalNB 실패
ClassifierChain 실패
ComplementNB 의 정답률: 0.9385964912280702
DecisionTreeClassifier 의 정답률: 0.9736842105263158
DummyClassifier 의 정답률: 0.6052631578947368
ExtraTreeClassifier 의 정답률: 0.9298245614035088
ExtraTreesClassifier 의 정답률: 0.9912280701754386
GaussianNB 의 정답률: 0.9473684210526315
GaussianProcessClassifier 의 정답률: 0.9473684210526315
GradientBoostingClassifier 의 정답률: 0.9912280701754386
HistGradientBoostingClassifier 의 정답률: 0.9912280701754386
KNeighborsClassifier 의 정답률: 0.9473684210526315
LabelPropagation 의 정답률: 0.41228070175438597
LabelSpreading 의 정답률: 0.41228070175438597
LinearDiscriminantAnalysis 의 정답률: 0.9824561403508771
LinearSVC 의 정답률: 0.9736842105263158
LogisticRegression 의 정답률: 0.9649122807017544
LogisticRegressionCV 의 정답률: 0.9649122807017544
MLPClassifier 의 정답률: 0.9649122807017544
MultiOutputClassifier 실패
MultinomialNB 의 정답률: 0.9385964912280702
NearestCentroid 의 정답률: 0.9210526315789473
NuSVC 의 정답률: 0.9035087719298246
OneVsOneClassifier 실패
OneVsRestClassifier 실패
OutputCodeClassifier 실패
PassiveAggressiveClassifier 의 정답률: 0.9385964912280702
Perceptron 의 정답률: 0.9473684210526315
QuadraticDiscriminantAnalysis 의 정답률: 0.9649122807017544
RadiusNeighborsClassifier 실패
RandomForestClassifier 의 정답률: 0.9736842105263158
RidgeClassifier 의 정답률: 0.9824561403508771
RidgeClassifierCV 의 정답률: 0.9824561403508771
SGDClassifier 의 정답률: 0.9473684210526315
SVC 의 정답률: 0.9649122807017544
StackingClassifier 실패
VotingClassifier 실패
'''
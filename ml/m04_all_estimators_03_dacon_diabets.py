from sklearn.metrics import r2_score, mean_squared_error, mean_squared_log_error
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense,LSTM,Conv1D,Flatten
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,f1_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import Perceptron,LogisticRegression
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
#1. 데이터

path= "c:\_data\dacon\cancer\\"
train_csv = pd.read_csv(path+"train.csv",index_col=0)
test_csv = pd.read_csv(path+"test.csv",index_col=0)
sampleSubmission_csv = pd.read_csv(path+"sample_Submission.csv")

# print("train",train_csv.shape)      #(652,9)
# print("test",test_csv.shape)       #(116, 8)
# print("sub",sampleSubmission_csv.shape) #(116,2)

x= train_csv.drop(['Outcome'], axis=1)
y= train_csv['Outcome']

x_train,x_test,y_train,y_test=train_test_split(x,y, train_size=0.9, random_state=8)
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
AdaBoostClassifier 의 정답률: 0.7727272727272727
BaggingClassifier 의 정답률: 0.7424242424242424
BernoulliNB 의 정답률: 0.6515151515151515
CalibratedClassifierCV 의 정답률: 0.7272727272727273
CategoricalNB 실패
ClassifierChain 실패
ComplementNB 의 정답률: 0.5909090909090909
DecisionTreeClassifier 의 정답률: 0.7272727272727273
DummyClassifier 의 정답률: 0.6515151515151515
ExtraTreeClassifier 의 정답률: 0.6363636363636364
ExtraTreesClassifier 의 정답률: 0.7727272727272727
GaussianNB 의 정답률: 0.7878787878787878
GaussianProcessClassifier 의 정답률: 0.5909090909090909
GradientBoostingClassifier 의 정답률: 0.7727272727272727
HistGradientBoostingClassifier 의 정답률: 0.8181818181818182
KNeighborsClassifier 의 정답률: 0.6515151515151515
LabelPropagation 의 정답률: 0.6212121212121212
LabelSpreading 의 정답률: 0.6212121212121212
LinearDiscriminantAnalysis 의 정답률: 0.7727272727272727
LinearSVC 의 정답률: 0.7121212121212122
LogisticRegression 의 정답률: 0.7575757575757576
LogisticRegressionCV 의 정답률: 0.7727272727272727
MLPClassifier 의 정답률: 0.6818181818181818
MultiOutputClassifier 실패
MultinomialNB 의 정답률: 0.6060606060606061
NearestCentroid 의 정답률: 0.6515151515151515
NuSVC 의 정답률: 0.7575757575757576
OneVsOneClassifier 실패
OneVsRestClassifier 실패
OutputCodeClassifier 실패
PassiveAggressiveClassifier 의 정답률: 0.696969696969697
Perceptron 의 정답률: 0.6515151515151515
QuadraticDiscriminantAnalysis 의 정답률: 0.7424242424242424
RadiusNeighborsClassifier 실패
RandomForestClassifier 의 정답률: 0.7878787878787878
RidgeClassifier 의 정답률: 0.7727272727272727
RidgeClassifierCV 의 정답률: 0.7727272727272727
SGDClassifier 의 정답률: 0.5151515151515151
SVC 의 정답률: 0.7878787878787878
StackingClassifier 실패
VotingClassifier 실패
'''
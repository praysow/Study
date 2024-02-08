from keras.models import Sequential
from keras.layers import Dense, Dropout, AveragePooling2D, Flatten, Conv2D,Conv1D,Flatten
from sklearn.metrics import accuracy_score,f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import Perceptron,LogisticRegression
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
#1. 데이터
path= "c:\_data\dacon\wine\\"
train_csv = pd.read_csv(path+"train.csv",index_col=0)
test_csv = pd.read_csv(path+"test.csv",index_col=0)
sampleSubmission_csv = pd.read_csv(path+"sample_Submission.csv")
x= train_csv.drop(['quality'], axis=1)
y= train_csv['quality']

lb=LabelEncoder()
lb.fit(x['type'])
x['type'] =lb.transform(x['type'])
test_csv['type'] =lb.transform(test_csv['type'])

x_train,x_test,y_train,y_test=train_test_split(x,y, train_size= 0.9193904973982694, random_state=1909,
                                            stratify=y)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
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
AdaBoostClassifier 의 정답률: 0.49324324324324326
BaggingClassifier 의 정답률: 0.6531531531531531
BernoulliNB 의 정답률: 0.46396396396396394
CalibratedClassifierCV 의 정답률: 0.5563063063063063
CategoricalNB 의 정답률: 0.46396396396396394
ClassifierChain 실패
ComplementNB 의 정답률: 0.46846846846846846
DecisionTreeClassifier 의 정답률: 0.5923423423423423
DummyClassifier 의 정답률: 0.4391891891891892
ExtraTreeClassifier 의 정답률: 0.6441441441441441
ExtraTreesClassifier 의 정답률: 0.7004504504504504
GaussianNB 의 정답률: 0.32882882882882886
GaussianProcessClassifier 의 정답률: 0.5720720720720721
GradientBoostingClassifier 의 정답률: 0.6283783783783784
HistGradientBoostingClassifier 의 정답률: 0.6756756756756757
KNeighborsClassifier 의 정답률: 0.5698198198198198
LabelPropagation 의 정답률: 0.5180180180180181
LabelSpreading 의 정답률: 0.5135135135135135
LinearDiscriminantAnalysis 의 정답률: 0.5653153153153153
LinearSVC 의 정답률: 0.5563063063063063
LogisticRegression 의 정답률: 0.5743243243243243
LogisticRegressionCV 의 정답률: 0.581081081081081
MLPClassifier 의 정답률: 0.5855855855855856
MultiOutputClassifier 실패
MultinomialNB 의 정답률: 0.4391891891891892
NearestCentroid 의 정답률: 0.12837837837837837
NuSVC 실패
OneVsOneClassifier 실패
OneVsRestClassifier 실패
OutputCodeClassifier 실패
PassiveAggressiveClassifier 의 정답률: 0.4864864864864865
Perceptron 의 정답률: 0.27702702702702703
QuadraticDiscriminantAnalysis 의 정답률: 0.509009009009009
RadiusNeighborsClassifier 의 정답률: 0.4617117117117117
RandomForestClassifier 의 정답률: 0.6891891891891891
RidgeClassifier 의 정답률: 0.5540540540540541
RidgeClassifierCV 의 정답률: 0.5495495495495496
SGDClassifier 의 정답률: 0.5518018018018018
SVC 의 정답률: 0.5540540540540541
StackingClassifier 실패
VotingClassifier 실패
'''
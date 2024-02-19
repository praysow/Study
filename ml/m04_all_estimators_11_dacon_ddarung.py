# https://dacon.io/competitions/open/235576/leaderboard
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import Perceptron,LogisticRegression
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import pandas as pd

#1. 데이터

path= "c:\_data\dacon\ddarung\\"

train_csv = pd.read_csv(path+"train.csv",index_col=0)
test_csv= pd.read_csv(path+"test.csv",index_col=0)
submission_csv= pd.read_csv(path+"submission.csv")

train_csv=train_csv.fillna(train_csv.mean())                         #test는 dropna를 하면 안되고 결측치를 변경해줘야한다
# train_csv=train_csv.fillna(0)
test_csv=test_csv.fillna(test_csv.mean())                         #test는 dropna를 하면 안되고 결측치를 변경해줘야한다
# test_csv=test_csv.fillna(0)

x= train_csv.drop(['count'],axis=1)
y= train_csv['count']

x_train,x_test,y_train,y_test=train_test_split(x,y, train_size=0.8, random_state=6)

#2. 모델구성
from sklearn.utils import all_estimators
import warnings
warnings.filterwarnings('ignore')
#2.모델구성
# allAlgorithms = all_estimators(type_filter='classifier')
allAlgorithms = all_estimators(type_filter='regressor')

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
ARDRegression 의 정답률: 0.6123809335385241
AdaBoostRegressor 의 정답률: 0.5200214565139767
BaggingRegressor 의 정답률: 0.7609997612986177
BayesianRidge 의 정답률: 0.6043235974714554
CCA 실패
DecisionTreeRegressor 의 정답률: 0.6737065524654907
DummyRegressor 의 정답률: -0.0020923530110168453
ElasticNet 의 정답률: 0.5886576527382242
ElasticNetCV 의 정답률: 0.5530312969275624
ExtraTreeRegressor 의 정답률: 0.5933712521377251
ExtraTreesRegressor 의 정답률: 0.8200860507727257
GammaRegressor 의 정답률: 0.48870740504665866
GaussianProcessRegressor 의 정답률: -1.6595552003894043
GradientBoostingRegressor 의 정답률: 0.7879331590032661
HistGradientBoostingRegressor 의 정답률: 0.8042565908011237
HuberRegressor 의 정답률: 0.5955731559000625
IsotonicRegression 실패
KNeighborsRegressor 의 정답률: 0.3222955497852805
KernelRidge 의 정답률: 0.6106536932641906
Lars 의 정답률: 0.6111335907585842
LarsCV 의 정답률: 0.6111335907585842
Lasso 의 정답률: 0.5962763341278503
LassoCV 의 정답률: 0.5785034977184933
LassoLars 의 정답률: 0.5962769242375423
LassoLarsCV 의 정답률: 0.6111335907585842
LassoLarsIC 의 정답률: 0.6111335907585842
LinearRegression 의 정답률: 0.6111335907585849
LinearSVR 의 정답률: -0.040355407384081055
MLPRegressor 의 정답률: 0.607386243402217
MultiOutputRegressor 실패
MultiTaskElasticNet 실패
MultiTaskElasticNetCV 실패
MultiTaskLasso 실패
MultiTaskLassoCV 실패
NuSVR 의 정답률: 0.07182247627586757
OrthogonalMatchingPursuit 의 정답률: 0.09668011412037847
OrthogonalMatchingPursuitCV 의 정답률: 0.5811589920964046
PLSCanonical 실패
PLSRegression 의 정답률: 0.6095099956606749
PassiveAggressiveRegressor 의 정답률: 0.14198156947732277
PoissonRegressor 의 정답률: -0.0020428858852612475
QuantileRegressor 실패
RANSACRegressor 의 정답률: 0.5262736095750933
RadiusNeighborsRegressor 실패
RandomForestRegressor 의 정답률: 0.7904416351618204
RegressorChain 실패
Ridge 의 정답률: 0.6091767802514223
RidgeCV 의 정답률: 0.6110693094531185
SGDRegressor 의 정답률: -4.2322415619915175e+25
SVR 의 정답률: 0.08108128329179665
StackingRegressor 실패
TheilSenRegressor 의 정답률: 0.5993021716319342
TransformedTargetRegressor 의 정답률: 0.6111335907585849
TweedieRegressor 의 정답률: 0.5818341650823844
VotingRegressor 실패
'''
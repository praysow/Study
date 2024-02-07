from sklearn.metrics import r2_score, mean_squared_error, mean_squared_log_error
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import Perceptron,LogisticRegression
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
#1. 데이터

path= "c:\_data\kaggle\\bike\\"
train_csv = pd.read_csv(path+"train.csv",index_col=0)
test_csv = pd.read_csv(path+"test.csv",index_col=0)
sampleSubmission_csv = pd.read_csv(path+"sampleSubmission.csv")

# print(train_csv)
# print(test_csv)
# print(submission_csv)

# print("train",train_csv.shape)      #(10886, 11)
# print("test",test_csv.shape)       #(6493, 8)
# print("sub",sampleSubmission_csv.shape) #(6493, 2)

#train_csv=train_csv.dropna()
# train_csv=train_csv.fillna(train_csv.mean())
# train_csv=train_csv.fillna(0)
# test_csv=test_csv.fillna(test_csv.mean())
#test_csv=test_csv.fillna(0)

x= train_csv.drop(['count','casual','registered'], axis=1)
y= train_csv['count']

x_train,x_test,y_train,y_test=train_test_split(x,y, train_size=0.8, random_state=3)

#2.모델구성
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
RDRegression 의 정답률: 0.26838454065706374
AdaBoostRegressor 의 정답률: 0.21844947436923812
BaggingRegressor 의 정답률: 0.24222188551985901
BayesianRidge 의 정답률: 0.26897188689161833
CCA 실패
DecisionTreeRegressor 의 정답률: -0.1432630345316066
DummyRegressor 의 정답률: -0.0003848287234036185
ElasticNet 의 정답률: 0.26797845454779845
ElasticNetCV 의 정답률: 0.2656188385361625
ExtraTreeRegressor 의 정답률: -0.13123902431163614
ExtraTreesRegressor 의 정답률: 0.21579720250848267
GammaRegressor 의 정답률: 0.1865954819271276
GaussianProcessRegressor 의 정답률: -0.276879727900073
GradientBoostingRegressor 의 정답률: 0.3201051821678873
HistGradientBoostingRegressor 의 정답률: 0.3496910922708474
HuberRegressor 의 정답률: 0.2472596290229082
IsotonicRegression 실패
KNeighborsRegressor 의 정답률: 0.18854396850870092
KernelRidge 의 정답률: 0.25569290804073574
Lars 의 정답률: 0.26898166576338445       
LarsCV 의 정답률: 0.2686832887318339      
Lasso 의 정답률: 0.26845591076682795
LassoCV 의 정답률: 0.2683954930378214    
LassoLars 의 정답률: 0.2684409149339174  
LassoLarsCV 의 정답률: 0.2686832887318339
LassoLarsIC 의 정답률: 0.26877344571466555    
LinearRegression 의 정답률: 0.2689816657633842
LinearSVR 의 정답률: 0.22306699425197707
MLPRegressor 의 정답률: 0.28861763511139804
MultiOutputRegressor 실패
MultiTaskElasticNet 실패
MultiTaskElasticNetCV 실패
MultiTaskLasso 실패
MultiTaskLassoCV 실패
NuSVR 의 정답률: 0.22104968035616057
OrthogonalMatchingPursuit 의 정답률: 0.09727716992784441
OrthogonalMatchingPursuitCV 의 정답률: 0.26757048769629266
PLSCanonical 실패
PLSRegression 의 정답률: 0.2615435657074955
PassiveAggressiveRegressor 의 정답률: -0.20222387312491108
PoissonRegressor 의 정답률: 0.28246561577554696
QuantileRegressor 실패
RANSACRegressor 의 정답률: -0.07260772582347741
RadiusNeighborsRegressor 의 정답률: -1.2643956076984272e+33
RandomForestRegressor 의 정답률: 0.2909278690262024
RegressorChain 실패
Ridge 의 정답률: 0.26898130213006965
RidgeCV 의 정답률: 0.268978255315803
SGDRegressor 의 정답률: -538815022850169.56
SVR 의 정답률: 0.2033597524835289
StackingRegressor 실패
TheilSenRegressor 의 정답률: 0.26226273121465304
TransformedTargetRegressor 의 정답률: 0.2689816657633842
TweedieRegressor 의 정답률: 0.26649423509557224
VotingRegressor 실패
'''

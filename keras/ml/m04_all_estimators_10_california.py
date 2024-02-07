#r2 0.55 ~0.6이상
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import Perceptron,LogisticRegression
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.datasets import fetch_california_housing
import time
#1.데이터
datasets = fetch_california_housing()
x =datasets.data
y =datasets.target

# print(x)   #(20640, 8)
# print(y)   #(20640,)
# print(x.shape,y.shape)
#print(datasets.feature_names)  #['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
import numpy as np

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.7, random_state=130)
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
ARDRegression 의 정답률: 0.5935869976295303
AdaBoostRegressor 의 정답률: 0.3870316130815761
BaggingRegressor 의 정답률: 0.7772318349836838
BayesianRidge 의 정답률: 0.6042520538216671
CCA 실패
DecisionTreeRegressor 의 정답률: 0.6027646737835677
DummyRegressor 의 정답률: -1.88647148473553e-05
ElasticNet 의 정답률: 0.42409051732883774
ElasticNetCV 의 정답률: 0.5870877572378987
ExtraTreeRegressor 의 정답률: 0.5359911720621541
ExtraTreesRegressor 의 정답률: 0.8040487985583671
GammaRegressor 의 정답률: -1.9077452078075652e-05
GaussianProcessRegressor 의 정답률: -2.865920159331365
GradientBoostingRegressor 의 정답률: 0.7810276631165207
HistGradientBoostingRegressor 의 정답률: 0.8283651533832025
HuberRegressor 의 정답률: 0.5236684632629252
IsotonicRegression 실패
KNeighborsRegressor 의 정답률: 0.12835844817304742
KernelRidge 의 정답률: 0.5468933190613987
Lars 의 정답률: 0.5427624689025738       
LarsCV 의 정답률: 0.5664395974720127
Lasso 의 정답률: 0.28579491882971786
LassoCV 의 정답률: 0.5906157088433139   
LassoLars 의 정답률: 0.28579393082655624
LassoLarsCV 의 정답률: 0.6013385868020502
LassoLarsIC 의 정답률: 0.6042486641813756
LinearRegression 의 정답률: 0.6042486641813757
LinearSVR 의 정답률: 0.4736608238129135
MLPRegressor 의 정답률: -0.448810357352843
MultiOutputRegressor 실패
MultiTaskElasticNet 실패
MultiTaskElasticNetCV 실패
MultiTaskLasso 실패
MultiTaskLassoCV 실패
NuSVR 의 정답률: 0.001596202682691228
OrthogonalMatchingPursuit 의 정답률: 0.0006141688997368666
OrthogonalMatchingPursuitCV 의 정답률: 0.5236067833047569
PLSCanonical 실패
PLSRegression 의 정답률: 0.5345472870982151
PassiveAggressiveRegressor 의 정답률: -1.7990234315482923
PoissonRegressor 의 정답률: 0.46021760079988605
QuantileRegressor 실패
RANSACRegressor 의 정답률: 0.41182749449714295
RadiusNeighborsRegressor 실패
RandomForestRegressor 의 정답률: 0.7956049033590062
RegressorChain 실패
Ridge 의 정답률: 0.6042496163571804
RidgeCV 의 정답률: 0.6042539421571732
SGDRegressor 의 정답률: -1.7908034604069636e+30
SVR 의 정답률: -0.03261596742647921
StackingRegressor 실패
TheilSenRegressor 의 정답률: 0.43385993823156566        
TransformedTargetRegressor 의 정답률: 0.6042486641813757
TweedieRegressor 의 정답률: 0.49709119176017547
VotingRegressor 실패
'''



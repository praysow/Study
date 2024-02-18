import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import Perceptron,LogisticRegression,SGDClassifier
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor,GradientBoostingRegressor
from sklearn.utils import all_estimators
import warnings
from sklearn.metrics import accuracy_score
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV,HalvingGridSearchCV,HalvingRandomSearchCV
warnings.filterwarnings('ignore')
import time
#1. 데이터

path= "c:\_data\kaggle\\bike\\"
train_csv = pd.read_csv(path+"train.csv",index_col=0)
test_csv = pd.read_csv(path+"test.csv",index_col=0)
sampleSubmission_csv = pd.read_csv(path+"sampleSubmission.csv")


x= train_csv.drop(['count'], axis=1)
y= train_csv['count']

# print(x.columns)
# print(x.shape)


x_train,x_test,y_train,y_test=train_test_split(x,y, train_size=0.8, random_state=3)
from xgboost import XGBRegressor


allAlgorithms = [
    ('LogisticRegression', XGBRegressor),
    ('KNeighborsClassifier',GradientBoostingRegressor),
    ('DecisionTreeClassifier', DecisionTreeRegressor),
    ('RandomForestClassifier', RandomForestRegressor)
]

''' 25퍼 미만 열 삭제 '''
# columns = datasets.feature_names
columns = x.columns
x = pd.DataFrame(x,columns=columns)
print("x.shape",x.shape)
''' 이 밑에 숫자에 얻은 feature_importances 넣고 줄 끝마다 \만 붙여주기'''
fi_str = "2.09285633e-05 5.38912208e-06 3.89393150e-05 1.21661375e-05\
 5.37875096e-05 4.63619818e-05 7.64672136e-05 6.26248892e-05\
 5.13916830e-02 9.48291652e-01"
 
''' str에서 숫자로 변환하는 구간 '''
fi_str = fi_str.split()
fi_float = [float(s) for s in fi_str]
print(fi_float)
fi_list = pd.Series(fi_float)

''' 25퍼 미만 인덱스 구하기 '''
low_idx_list = fi_list[fi_list <= fi_list.quantile(0.25)].index
print('low_idx_list',low_idx_list)

''' 25퍼 미만 제거하기 '''
low_col_list = [x.columns[index] for index in low_idx_list]
# 이건 혹여 중복되는 값들이 많아 25퍼이상으로 넘어갈시 25퍼로 자르기
if len(low_col_list) > len(x.columns) * 0.25:   
    low_col_list = low_col_list[:int(len(x.columns)*0.25)]
print('low_col_list',low_col_list)
x.drop(low_col_list,axis=1,inplace=True)
print("after x.shape",x.shape)

# 3. 모델 훈련 및 평가
for name, algorithm in allAlgorithms:
    model = algorithm()
    model.fit(x_train, y_train)
    acc = model.score(x_test, y_test)
    print(name, '의 정확도:', acc)

# print(model.feature_importances_)
    
'''
LogisticRegression 의 정확도: 0.9990321405083594
KNeighborsClassifier 의 정확도: 0.9990407703597125
DecisionTreeClassifier 의 정확도: 0.9985899373664799
RandomForestClassifier 의 정확도: 0.9994766030095583
'''
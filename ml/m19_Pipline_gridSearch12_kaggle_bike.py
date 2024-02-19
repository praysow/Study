import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import Perceptron,LogisticRegression,SGDClassifier
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.utils import all_estimators
import warnings
from sklearn.metrics import accuracy_score
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV,HalvingGridSearchCV
warnings.filterwarnings('ignore')
import time
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
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer, RobustScaler
scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
parameters = [
    {'RF__n_estimators' : [100,200], 'RF__max_depth':[6,10,12],'RF__min_samples_leaf' : [3,10]},
    {'RF__max_depth' : [6,8,10,12], 'RF__min_samples_leaf':[3,5,7,10]},
    {'RF__min_samples_leaf':[3,5,7,10],'RF__min_samples_split':[2,3,5,10]},
    {'RF__min_samples_split' : [2,3,5,10]},
    {'RF__n_jobs':[-1,2,4], 'RF__min_samples_split' : [2,3,5,10]}
] 
from sklearn.preprocessing import RobustScaler,MinMaxScaler,StandardScaler,MaxAbsScaler
from sklearn.pipeline import make_pipeline,Pipeline
pipe = Pipeline([('MinMax',MinMaxScaler()),('RF',RandomForestRegressor())])
model1 = GridSearchCV(pipe,parameters,cv=5,verbose=1)
model2 = RandomizedSearchCV(pipe,parameters,cv=5,verbose=1)
model3 = HalvingGridSearchCV(pipe,parameters,cv=5,verbose=1)
model = model3
model.fit(x_train,y_train)

# 4.결과예측
result = model.score(x_test,y_test)
print("acc :", result)
'''

'''
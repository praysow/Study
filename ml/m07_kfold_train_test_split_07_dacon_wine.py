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
from lightgbm import LGBMClassifier

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

from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer, RobustScaler
scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import StratifiedKFold,cross_val_predict
n_split = 5
kfold = KFold(n_splits=n_split,shuffle=True, random_state=123)
# from sklearn.model_selection import StratifiedKFold
# kfold = StratifiedKFold(n_splits=n_split,shuffle=True, random_state=123)

#2.모델
model = SVC()   #소프트벡터머신 클래스파이어
#3.훈련
scores = cross_val_score(model, x_train, y_train, cv=kfold)
print("ACC:",scores,"\n평균:",round(np.mean(scores),4))
from sklearn.metrics import accuracy_score
y_pred = cross_val_predict(model, x_test, y_test, cv=kfold)
acc = accuracy_score(y_test, y_pred)
print("acc", acc)

'''
ACC: [0.62818182 0.65363636 0.63876251 0.633303   0.63057325] 
 평균: 0.6369
'''
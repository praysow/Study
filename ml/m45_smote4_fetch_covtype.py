import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import fetch_covtype

#1.데이터
datasets= fetch_covtype()
x= datasets.data
y= datasets.target


x = x[:-30]
y = y[:-30]
# print(y)
# print(np.unique(y, return_counts=True))
# print(x.shape,y.shape)

x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.9,random_state=3,stratify=y)   #stratify=y의 비율대로 잘라라

from imblearn.over_sampling import SMOTE
import sklearn as sk
# print("sk",sk.__version__)

smote= SMOTE(random_state=1)
x_train,y_train = smote.fit_resample(x_train,y_train)
# print(x_train.shape,y_train.shape)

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

model = RandomForestClassifier()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

r2 = accuracy_score(y_test, y_pred)
print("R2 Score:", r2)

'''
R2 Score: 0.962064751544777
'''
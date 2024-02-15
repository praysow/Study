#y값이 필요없기때문에 대표적인 비지도 학습중하나이다
#데이터에 0이 많은 것들은 성능이 더 좋게 나올수도 있다.
#보통 PCA를 하기전에 스탠다드스케일러를 사용하면 좋다
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import sklearn as sk
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor,GradientBoostingClassifier

#1. 데이터
datasets = load_iris()
x = datasets.data
y = datasets.target

scaler = StandardScaler()
x = scaler.fit_transform(x)

pca = PCA(n_components=1)
x= pca.fit_transform(x)
print(x.shape)
x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.7,random_state=3,shuffle=True,stratify=y)

model = RandomForestClassifier(random_state=777)
model.fit(x_train,y_train)

result = model.score(x_test,y_test)
print("score : ",result)
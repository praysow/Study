import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical


#1.데이터
#원핫1.케라스
datasets = load_iris()
x = datasets.data
y = datasets.target
y_ohe = to_categorical(datasets.target)

print(y_ohe.shape)      #(150,3)
# print(y_ohe)

#원핫2.판다스
y_ohe2 = pd.get_dummies(y)
print(y_ohe2.shape)
# print(y_ohe2)

#원핫3. 사이킷런
from sklearn.preprocessing import OneHotEncoder
# y = y.reshape(-3,1)        #(150,1)//사이킥런에서는 행렬형태로 바꿔야 작동한다


print(y.shape)

y = y.reshape(-1,1)        #(150,1)// 백터를 행렬의 형태로 바꾸기 대괄호가 하나 있던것을 대괄호를 3개로 바꾸는 것
print(y.shape)
print(x.shape)
# y = y.reshape(150,1)         #(150,1)//reshape에서 데이터의 순서와 갯수만 달라지지 않는다면 언제든지 자를수 있다(분할)

# OHE = OneHotEncoder(sparse=False).fit(y_ohe3)
# y_ohe3 = OHE.transform(y_ohe3)
'''
OHE = OneHotEncoder(sparse=False)#디폴트는 True
OHE = OneHotEncoder()
# OHE.fit(y)                     #import>fit>transform
# y_ohe3 = OHE.transform(y)      #fit으로 저장을하고 transform으로 바꾼다
y_ohe3 = OHE.fit_transform(y)    # 간단하것으로 바꾸는 것
y_ohe3 = OHE.fit_transform(y).toarray()    # toarray를 사용하면 (150,3)에서 (150,1)로 바꿈

print(y_ohe3)
print(y_ohe3.shape)
'''
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import pandas as pd
from keras.models import Sequential, Model
from keras.layers import Dense,Dropout, LSTM,Conv1D,Flatten
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
#1. 데이터
datasets= load_breast_cancer()

x = datasets.data       #(569, 30)
y = datasets.target     #(569,)
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8, random_state=450)

#3.컴파일 훈련
from sklearn.svm import LinearSVC
model = LinearSVC(C=100)

model.fit(x_train,y_train)

# 4.결과예측
result = model.score(x_test,y_test)
print("acc :", result)
y_predict = model.predict(x_test)
# print(y_predict)

'''

로스: 0.034642551094293594
R2 score 0.9536113794489123
accuracy : 0.9824561476707458

로스: 0.0669967532157898
R2 score 0.9115674319991807
accuracy : 0.9649122953414917
acc : 0.9649122807017544
'''
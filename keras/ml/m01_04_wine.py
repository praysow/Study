import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM,Conv1D,Flatten
from keras.utils import to_categorical
from sklearn.metrics import f1_score

#1.데이터
datasets= load_wine()
x= datasets.data
y= datasets.target

# # 사이킷런

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8,
                                                    random_state=383,        
                                                    # stratify=y_ohe1            
                                                    )

from sklearn.svm import LinearSVC       #SVC다중 클래스파이어 SVR회귀모델
model = LinearSVC()

model.fit(x_train,y_train)

# 4.결과예측
result = model.score(x_test,y_test)
print("acc :", result)
y_predict = model.predict(x_test)
print(y_predict)
f1=f1_score(y_test,y_predict, average='macro')


print("f1",f1)
print("로스:", result)

'''
accuracy_score : 0.9444444444444444
로스 : 0.11515446752309799
acc : 0.9444444179534912

accuracy_score : 1.0
로스 : 0.015490112826228142
acc : 1.0                       아래두개

accuracy_score : 0.9444444444444444
로스 : 0.2360297292470932
acc : 0.9444444179534912

accuracy_score : 0.9722222222222222
로스 : 0.060165151953697205
acc : 0.9722222089767456
'''
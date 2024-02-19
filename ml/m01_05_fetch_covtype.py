import numpy as np
import pandas as pd
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers import Dense,LSTM,Conv1D,Flatten
from keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import f1_score
import time
#1.데이터
datasets= fetch_covtype()
x= datasets.data
y= datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.86,
                                                    random_state=5,        #346
                                                    # stratify=y_ohe1            
                                                    )

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
# scaler = MinMaxScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)

from sklearn.svm import LinearSVC
model = LinearSVC()

model.fit(x_train,y_train)

# 4.결과예측
result = model.score(x_test,y_test)
print("acc :", result)
y_predict = model.predict(x_test)
# print(y_predict)
f1=f1_score(y_test,y_predict, average='macro')


print("f1",f1)
print("로스:", result)
'''

accuracy_score : 0.5265053723783532
로스 : 1.044301986694336
acc : 0.5265053510665894
걸린시간 : 506

accuracy_score : 0.7191856605443683
로스 : 0.6707912087440491
acc : 0.7191856503486633
걸린시간 : 83
'''
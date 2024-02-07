from keras.models import Sequential
from keras.layers import Dense, Dropout, AveragePooling2D, Flatten, Conv2D,Conv1D,Flatten
from sklearn.metrics import accuracy_score,f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd

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

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

from sklearn.svm import LinearSVC
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
ACC : 0.5833333333333334
로스 : [1.0416243076324463, 0.5833333134651184]         load


ACC : 0.6216216216216216
로스 : [0.9659953117370605, 0.6216216087341309]

ACC : 0.6058558558558559
로스 : [1.0000303983688354, 0.6058558821678162]
'''
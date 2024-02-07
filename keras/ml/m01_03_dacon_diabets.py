from sklearn.metrics import r2_score, mean_squared_error, mean_squared_log_error
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense,LSTM,Conv1D,Flatten
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,f1_score
from sklearn.preprocessing import OneHotEncoder
#1. 데이터

path= "c:\_data\dacon\cancer\\"
train_csv = pd.read_csv(path+"train.csv",index_col=0)
test_csv = pd.read_csv(path+"test.csv",index_col=0)
sampleSubmission_csv = pd.read_csv(path+"sample_Submission.csv")

# print("train",train_csv.shape)      #(652,9)
# print("test",test_csv.shape)       #(116, 8)
# print("sub",sampleSubmission_csv.shape) #(116,2)

x= train_csv.drop(['Outcome'], axis=1)
y= train_csv['Outcome']

x_train,x_test,y_train,y_test=train_test_split(x,y, train_size=0.9, random_state=8)
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
ACC : 0.7727272727272727
로스 : 0.6435238718986511

ACC : 0.7424242424242424
로스 : 0.5378472805023193
'''
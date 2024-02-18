from sklearn.metrics import r2_score, mean_squared_error, mean_squared_log_error
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import pandas as pd

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

#2.모델구성
from sklearn.svm import LinearSVR
model = LinearSVR()

model.fit(x_train,y_train)

# 4.결과예측
result = model.score(x_test,y_test)
y_predict = model.predict(x_test)
from sklearn.metrics import r2_score
r2=r2_score(y_test,y_predict)
y_submit = model.predict(test_csv)
sampleSubmission_csv['count'] = y_submit
sampleSubmission_csv.to_csv(path + "submission_21.csv", index=False)
# print(submission_csv)
print("R2 score",r2)
print("acc :", result)
'''

'''

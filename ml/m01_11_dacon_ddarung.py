# https://dacon.io/competitions/open/235576/leaderboard

from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import pandas as pd

#1. 데이터

path= "c:\_data\dacon\ddarung\\"

train_csv = pd.read_csv(path+"train.csv",index_col=0)
test_csv= pd.read_csv(path+"test.csv",index_col=0)
submission_csv= pd.read_csv(path+"submission.csv")

train_csv=train_csv.fillna(train_csv.mean())                         #test는 dropna를 하면 안되고 결측치를 변경해줘야한다
# train_csv=train_csv.fillna(0)
test_csv=test_csv.fillna(test_csv.mean())                         #test는 dropna를 하면 안되고 결측치를 변경해줘야한다
# test_csv=test_csv.fillna(0)

x= train_csv.drop(['count'],axis=1)
y= train_csv['count']

x_train,x_test,y_train,y_test=train_test_split(x,y, train_size=0.8, random_state=6)

#2. 모델구성
from sklearn.svm import LinearSVR
model = LinearSVR()

model.fit(x_train,y_train)

# 4.결과예측
result = model.score(x_test,y_test)
y_predict = model.predict(x_test)
from sklearn.metrics import r2_score
r2=r2_score(y_test,y_predict)
y_submit = model.predict(test_csv)
submission_csv['count'] = y_submit
submission_csv.to_csv(path + "submission_21.csv", index=False)
# print(submission_csv)
print("R2 score",r2)
print("acc :", result)

'''

submission_16 로스 : 2469.941162109375  x,y, train_size=0.8, random_state=4
                                        10,1  epochs=2100, batch_size=32
submission_17 로스 : 2467.4755859375  x,y, train_size=0.8, random_state=4
                                        11,1  epochs=2100, batch_size=32

submission_18 로스 : 2458.29931640625 x,y, train_size=0.8, random_state=4
                                        13,1  epochs=2100, batch_size=32

submission_18 로스 : 2450.8515625 x,y, train_size=0.8, random_state=4
                                        13,1  epochs=1900, batch_size=32

submission_19 로스 : 2441.406005859375 train_size=0.8, random_state=4
                                        13,1  epochs=1910, batch_size=30

submission_20 로스 : 2429.23046875 train_size=0.8, random_state=4
                                        13,1  epochs=1911, batch_size=30



'''
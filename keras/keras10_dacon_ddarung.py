# https://dacon.io/competitions/open/235576/leaderboard

from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import pandas as pd

#1. 데이터

path= "c:\_data\dacon\ddarung\\"
# print(path+ "aaa.csv")      #c:\_data\dacon\ddarung\aaa.csv
# train_csv = pd.read_csv("c:/_data/dacon/ddarung//train.csv")
# test_csv= pd.read_csv("c:/_data/dacon/ddarung//test.csv")
# submission_scv= pd.read_csv("c:/_data/dacon/ddarung//submission.csv")
train_csv = pd.read_csv(path+"train.csv",index_col=0)
test_csv= pd.read_csv(path+"test.csv",index_col=0)
submission_csv= pd.read_csv(path+"submission.csv")
# print(train_csv)
# print(test_csv)
# print(submission_scv)

# print("train",train_csv.shape)      #(1459, 10)
# print("test",test_csv.shape)       #(715, 9)
# print("sub",submission_scv.shape) #(715, 2)

# print(train_csv.columns)
# 'id', 'hour', 'hour_bef_temperature', 'hour_bef_precipitation',
#        'hour_bef_windspeed', 'hour_bef_humidity', 'hour_bef_visibility',
#        'hour_bef_ozone', 'hour_bef_pm10', 'hour_bef_pm2.5', 'count'],
#       dtype='object')
# print(train_csv.info())
# print(train_csv.describe()) 최대값 확인하는 법
# datasets= path
# train_csv = train_csv.dropna()
train_csv=train_csv.fillna(train_csv.mean())                         #test는 dropna를 하면 안되고 결측치를 변경해줘야한다
# train_csv=train_csv.fillna(0)
test_csv=test_csv.fillna(test_csv.mean())                         #test는 dropna를 하면 안되고 결측치를 변경해줘야한다
# test_csv=test_csv.fillna(0)


#######x와 y분리#######
x= train_csv.drop(['count'],axis=1)
y= train_csv['count']


#####결측치확인######
# print(train_csv.isna().sum())         isna 어디에 nan값이 있나
# print(train_csv.isna().sum())
# print(train_csv.info())
# print(train_csv.shape)      (1328, 10)


x_train,x_test,y_train,y_test=train_test_split(x,y, train_size=0.8, random_state=6)

#2. 모델구성
model=Sequential()
model.add(Dense(15,input_dim=9))
# model.add(Dense(20))
# model.add(Dense(100))
# model.add(Dense(150))
# model.add(Dense(80))
# model.add(Dense(40))
# model.add(Dense(10))
model.add(Dense(1))



#3.컴파일,훈련
model.compile(loss='mse',optimizer='adam')
model.fit(x_train, y_train, epochs=1915, batch_size=30)

#4.결과예측
loss = model.evaluate(x_test,y_test)
y_submit=model.predict(test_csv)
y_pred = model.predict(x_test)
print("로스 :",loss)

#submission.csv 만들기 count컬럼에 값만 넣어주기
submission_csv['count'] = y_submit
print(submission_csv)
submission_csv.to_csv(path + "submission_21.csv", index=False)


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
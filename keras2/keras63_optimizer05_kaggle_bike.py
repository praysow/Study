from sklearn.metrics import r2_score, mean_squared_error, mean_squared_log_error
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout
import numpy as np
import pandas as pd

#1. 데이터

path= "c:\_data\kaggle\\bike\\"
train_csv = pd.read_csv(path+"train.csv",index_col=0)
test_csv = pd.read_csv(path+"test.csv",index_col=0)
sampleSubmission_csv = pd.read_csv(path+"sampleSubmission.csv")

x= train_csv.drop(['count','casual','registered'], axis=1)
y= train_csv['count']

x_train,x_test,y_train,y_test=train_test_split(x,y, train_size=0.8, random_state=3)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
scaler = MaxAbsScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2.모델구성
model=Sequential()
model.add(Dense(10,input_dim=8,activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(10,activation='relu'))
model.add(Dense(1,activation='relu'))
model.add(Dense(20,activation='relu'))
model.add(Dense(50,activation='relu'))
model.add(Dense(200,activation='relu'))
model.add(Dense(50,activation='relu'))
model.add(Dense(20,activation='relu'))
model.add(Dense(1,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(1))


#3.컴파일 훈련
from keras.optimizers import Adam
learning_rate = 0.01
model.compile(loss='mse',optimizer=Adam(learning_rate=learning_rate))
hist= model.fit(x_train, y_train, epochs=100,batch_size=1000, validation_split=0.1,verbose=2,)

model.save("c:\_data\_save\kaggle_bike_1.h5")

#4.결과예측
loss=model.evaluate(x_test,y_test)
y_submit=model.predict(test_csv)
print("로스 :",loss)
print("음수 개수:",sampleSubmission_csv[sampleSubmission_csv['count']<0].count())

y_pred= model.predict(x_test)
r2=r2_score(y_test,y_pred)
print("R2 score",r2)
print("lr:{0},acc:{1}".format(learning_rate,r2))
print("로스 :",loss)
'''
lr:0.1,acc:-0.42031840478934623
로스 : 46025.6328125
lr:0.01,acc:-1.0090311337760225
로스 : 65102.95703125
'''
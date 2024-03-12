from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense,Dropout
import numpy as np
import pandas as pd

#1. 데이터

path= "c:\_data\dacon\ddarung\\"
train_csv = pd.read_csv(path+"train.csv",index_col=0)
test_csv= pd.read_csv(path+"test.csv",index_col=0)
submission_csv= pd.read_csv(path+"submission.csv")

# train_csv = train_csv.dropna()
train_csv=train_csv.fillna(train_csv.mean())                         #test는 dropna를 하면 안되고 결측치를 변경해줘야한다
# train_csv=train_csv.fillna(0)
test_csv=test_csv.fillna(test_csv.mean())                         #test는 dropna를 하면 안되고 결측치를 변경해줘야한다
# test_csv=test_csv.fillna(0)
x= train_csv.drop(['count'],axis=1)
y= train_csv['count']

x_train,x_test,y_train,y_test=train_test_split(x,y, train_size=0.8, random_state=6)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
scaler = MaxAbsScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델구성
model=Sequential()
model.add(Dense(15,input_dim=9,activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(20,activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(100,activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(150,activation='relu'))
model.add(Dense(80,activation='relu'))
model.add(Dense(40,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(1))



#3.컴파일,훈련
from keras.optimizers import Adam
learning_rate= 0.001
model.compile(loss='mse',optimizer=Adam(learning_rate=learning_rate))
hist= model.fit(x_train, y_train, epochs=100,batch_size=1000, validation_split=0.1,verbose=2)

#4.결과예측
loss = model.evaluate(x_test,y_test)
y_submit=model.predict(test_csv)

submission_csv['count'] = y_submit
# print(submission_csv)
submission_csv.to_csv(path + "submission_21.csv", index=False)
print("로스 :",loss)

from sklearn.metrics import r2_score,accuracy_score
y_pred = model.predict(x_test)
r2=r2_score(y_test,y_pred)
print("R2 score",r2)
print("lr:{0},acc:{1}".format(learning_rate,r2))
print("로스 :",loss)
'''
lr:0.1,acc:-1.4683329067125008
로스 : 16501.005859375
lr:0.01,acc:0.6301701988662183
로스 : 2472.34228515625
lr:0.001,acc:0.6107397385138346
로스 : 2602.23583984375
'''
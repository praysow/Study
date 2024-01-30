from sklearn.metrics import r2_score, mean_squared_error, mean_squared_log_error
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM
import numpy as np
import pandas as pd
from keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder


#1. 데이터

path= "c:\_data\kaggle\\bike\\"
train_csv = pd.read_csv(path+"train.csv",index_col=0)
test_csv = pd.read_csv(path+"test.csv",index_col=0)
sampleSubmission_csv = pd.read_csv(path+"sampleSubmission.csv")

x= train_csv.drop(['count','casual','registered'], axis=1)
y= train_csv['count']
x= x.astype('float32')

y=y.values.reshape(-1,1)

ohe = OneHotEncoder(sparse=False)
ohe = OneHotEncoder()
y= ohe.fit_transform(y).toarray()

x_train,x_test,y_train,y_test=train_test_split(x,y, train_size=0.8, random_state=3)

x_train = x_train.values.reshape(x_train.shape[0],8,1)
x_test = x_test.values.reshape(x_test.shape[0],8,1)
test_csv = test_csv.values.reshape(test_csv.shape[0],8,1)
# y_test=to_categorical(y_test)
# y_train=to_categorical(y_train)

# y=y.reshape(-1,1)

# ohe = OneHotEncoder(sparse=False)
# ohe = OneHotEncoder()
# y_ohe3= ohe.fit_transform(y).toarray()
# print(np.unique(x_train,return_counts=True))
# print(pd.value_counts(x_test))




#2.모델구성
model=Sequential()
model.add(LSTM(32,input_shape=(8,1),activation='relu'))
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
model.add(Dense(822,activation='softmax'))


#3.컴파일 훈련
from keras.callbacks import EarlyStopping, ModelCheckpoint

import datetime
date= datetime.datetime.now()
print(date)     
date = date.strftime("%m-%d_%H-%M")
print(date) 
print(type(date)) 

path='..//_data//_save//MCP/k27/'
filename= "{epoch:04d}-{val_loss:.4f}.hdf5"  
filepath = "".join([path,'05_kaggle_bike_',date,'_',filename])

mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    verbose=1,
    save_best_only=True,
    filepath=filepath
    )
es= EarlyStopping(monitor='val_loss',mode='min',patience=100,verbose=1,restore_best_weights=True)
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics='accuracy')
hist= model.fit(x_train, y_train, epochs=1000,batch_size=1000, validation_split=0.1,verbose=2,
          callbacks=[es,mcp])

model.save("c:\_data\_save\kaggle_bike_1.h5")

#4.결과예측
loss=model.evaluate(x_test,y_test)
y_submit=model.predict(test_csv)

print("로스 :",loss)
print("음수 개수:",sampleSubmission_csv[sampleSubmission_csv['count']<0].count())

# y_predict= model.predict(x_test)

# def RMSE(y_test, y_predict):
#     np.sqrt(mean_squared_error(y_test,y_predict))
#     return np.sqrt(mean_squared_error(y_test,y_predict))
# rmse=RMSE(y_test,y_predict)
# print("RMSE :",rmse)

# def RMSLE(y_test, y_predict):
#     # np.sqrt(mean_squared_error(y_test,y_predict))
#     return np.sqrt(mean_squared_log_error(y_test,y_predict))
# rmsle=RMSLE(y_test,y_predict)
# print("RMSLE :", rmsle)

# 로스 : 32414.482421875
# 음수 개수: datetime    0
# count       0
# dtype: int64
# 69/69 [==============================] - 0s 524us/step
# RMSLE : 1.5823909639816276

# [6493 rows x 2 columns]
# 로스 : 22648.681640625
# 음수 개수: datetime    0
# count       0
# dtype: int64
# 69/69 [==============================] - 0s 383us/step
# RMSLE : 1.3435691723506198                  위에하나 15번

# [6493 rows x 2 columns]
# 로스 : 22528.884765625
# 음수 개수: datetime    0
# count       0
# dtype: int64
# 69/69 [==============================] - 0s 635us/step
# RMSLE : 1.332195783025249

# 로스 : [6.092616558074951, 0.00826446246355772]
# 음수 개수: datetime    0
# count       0
# dtype: int64
from sklearn.metrics import r2_score, mean_squared_error, mean_squared_log_error
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM,Conv1D,Flatten
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
# model=Sequential()
# model.add(Conv1D(filters=32,kernel_size=2,input_shape=(8,1),activation='relu'))
# model.add(Flatten())
# model.add(Dense(10,activation='relu'))
# model.add(Dense(1,activation='relu'))
# model.add(Dense(20,activation='relu'))
# model.add(Dense(50,activation='relu'))
# model.add(Dense(200,activation='relu'))
# model.add(Dense(50,activation='relu'))
# model.add(Dense(20,activation='relu'))
# model.add(Dense(1,activation='relu'))
# model.add(Dense(10,activation='relu'))
# model.add(Dense(10,activation='relu'))
# model.add(Dense(822,activation='softmax'))

model=Sequential()
model.add(Conv1D(filters=32, kernel_size=2, input_shape=(8,1)))
model.add(Flatten())
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(1, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(822, activation='softmax'))

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
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
hist= model.fit(x_train, y_train, epochs=100,batch_size=1000, validation_split=0.1,verbose=2,
          callbacks=[es,mcp])

model.save("c:\_data\_save\kaggle_bike_1.h5")

#4.결과예측
loss=model.evaluate(x_test,y_test)
y_submit=model.predict(test_csv)
y_test = np.argmax(y_test,axis=1)
y_predict= np.argmax(y_submit,axis=1)
print("로스 :",loss)
print("음수 개수:",sampleSubmission_csv[sampleSubmission_csv['count']<0].count())


# 로스 : [6.092616558074951, 0.00826446246355772]
# 음수 개수: datetime    0
# count       0
# dtype: int64

# 로스 : [6.388578414916992, 0.00826446246355772]
# 음수 개수: datetime    0
# count       0
# dtype: int64
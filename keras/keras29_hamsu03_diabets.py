#R2 0.62이상
from sklearn.datasets import load_diabetes

#1.데이터
datasets= load_diabetes()
x= datasets.data
y= datasets.target

# print(x.shape) #(442,10)
# print(y.shape) #(442,)

# print(datasets.feature_names)
# print(datasets.DESCR)
from sklearn.model_selection import train_test_split
from keras.models import Sequential,Model
from keras.layers import Dense,Dropout,Input
import numpy as np

x_train,x_test,y_train,y_test= train_test_split(x,y,train_size=0.9, random_state=10)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
scaler = MaxAbsScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2.모델구성
# model=Sequential()
# model.add(Dense(1,input_dim=10))
# model.add(Dropout(0.1))
# model.add(Dense(10))
# model.add(Dense(100))
# model.add(Dense(10))
# model.add(Dense(1))
# model.add(Dense(10))
# model.add(Dense(100))
# model.add(Dense(10))
# model.add(Dense(1))

input1= Input(shape=(10,))
dense1= Dense(1)(input1)
drop1=Dropout(0.1)(dense1)
dense2= Dense(10)(dense1)
dense3= Dense(100)(dense2)
dense4= Dense(10)(dense3)
dense5= Dense(1)(dense4)
dense6= Dense(10)(dense5)
dense7= Dense(100)(dense6)
dense8= Dense(10)(dense7)
output= Dense(1)(dense8)
model = Model(inputs=input1,outputs=output)


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
filepath = "".join([path,'03_diabets_',date,'_',filename])

mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    verbose=1,
    save_best_only=True,
    filepath=filepath
    )
es= EarlyStopping(monitor='val_loss',mode='min',patience=100,verbose=1,restore_best_weights=True)
model.compile(loss='mse',optimizer='adam')
hist= model.fit(x_train, y_train, epochs=1000,batch_size=1000, validation_split=0.1,verbose=2,
          callbacks=[es,mcp])

model.save("c:\_data\_save\diabets_1.h5")

#4.결과예측
loss=model.evaluate(x_test,y_test)
y_predict=model.predict([x_test])
result=model.predict(x)
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score
r2=r2_score(y_test,y_predict)
print("로스 ,",loss)
print("R2 score :",r2)

'''
로스 , 2246.01953125
R2 score : 0.6280751199789454

로스 , 2249.0537109375
R2 score : 0.6275726473646075       위에 하나
'''
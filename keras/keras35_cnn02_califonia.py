from sklearn.datasets import fetch_california_housing
import time
import pandas as pd
#1.데이터
datasets = fetch_california_housing()
x =datasets.data
y =datasets.target

# print(x)   #(20640, 8)
# print(y)   #(20640,)
# print(x.shape,y.shape)

x=x.reshape(x.shape[0],4,2,1)


#print(datasets.feature_names)  #['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers import Dense,Dropout,Input,Conv2D,MaxPooling2D,Flatten
from keras.utils import to_categorical
import numpy as np

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.7, random_state=130)

# from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
# scaler = MaxAbsScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)
# x_train=x_train.reshape(x_train.shape[0],4,2,1)
# x_test=x_test.reshape(x_test.shape[0],4,2,1)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# print(pd.value_counts(x_train))


#2. 모델구성
model=Sequential()
model.add(Conv2D(32, (2,1),input_shape=(4,2,1),activation='relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.1))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(1))
model.add(Dense(100))
model.add(Dense(1))
model.add(Dense(100))
model.add(Dense(6,activation='softmax'))

# input1= Input(shape=(8,))
# dense1= Dense(1)(input1)
# drop1=Dropout(0.1)(dense1)
# dense2= Dense(100)(dense1)
# dense3= Dense(1)(dense2)
# dense4= Dense(100)(dense3)
# dense5= Dense(1)(dense4)
# dense6= Dense(100)(dense5)
# output= Dense(1)(dense6)
# model = Model(inputs=input1,outputs=output)

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
filepath = "".join([path,'02_califonia_',date,'_',filename])

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

model.save("c:\_data\_save\caliponia_1.h5")

#4.결과예측
loss=model.evaluate(x_test,y_test)
y_predict=model.predict([x_test])
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score
r2=r2_score(y_test,y_predict)
print("로스:",loss)
print("R2 score",r2)

'''
로스: 0.5679296255111694
R2 score 0.5717274569573302

로스: 0.5642915368080139
R2 score 0.5744709251732929     아래 두줄

로스: 0.6440656185150146
R2 score 0.514313570945987

로스: [1.5045673847198486, 0.4075569212436676]
R2 score 0.0031915523895632383
'''
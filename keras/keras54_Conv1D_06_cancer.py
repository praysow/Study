import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import pandas as pd
from keras.models import Sequential, Model
from keras.layers import Dense,Dropout, LSTM,Conv1D,Flatten
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
#1. 데이터
datasets= load_breast_cancer()
# print(datasets)
# print(datasets.DESCR)
#print(datasets.feature_names)
x = datasets.data       #(569, 30)
y = datasets.target     #(569,)
# print(np.unique(y, return_counts=True))
# (array([0, 1]), array([212, 357], dtype=int64))
# print(x.shape,y.shape)
# print(pd.Series.unique(y))
# print(pd.Series.value_counts(y))
# print(pd.DataFrame(y).value_counts())
# print(pd.value_counts(y))

x=x.reshape(x.shape[0],10,3)
y=y.reshape(-1,1)

ohe = OneHotEncoder(sparse=False)
ohe = OneHotEncoder()
y= ohe.fit_transform(y).toarray()
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8, random_state=450)

x_train=np.asarray(x_train).astype(np.float32)
x_test=np.asarray(x_test).astype(np.float32)

# from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
# scaler = MaxAbsScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)

#2.모델구성
model=Sequential()
model.add(Conv1D(filters=32,kernel_size=2,input_shape=(10,3),activation='relu'))
model.add(Dropout(0.1))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(110))
model.add(Dense(120))
model.add(Dense(130))
model.add(Dense(2, activation='softmax'))

# input1= Input(shape=(30,))
# dense1= Dense(10)(input1)
# drop1=Dropout(0.1)(dense1)
# dense2= Dense(100)(dense1)
# dense3= Dense(110)(dense2)
# dense4= Dense(120)(dense3)
# dense5= Dense(130)(dense4)
# output= Dense(1)(dense5)
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
filepath = "".join([path,'06cancer_',date,'_',filename])

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

model.save("c:\_data\_save\cancer_1.h5")

#4.결과예측
loss,accuracy=model.evaluate(x_test,y_test)
y_predcit=model.predict([x_test])
result=model.predict(x)

import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
r2=r2_score(y_test,y_predcit)
print("로스:", loss)
print("R2 score",r2)
print("accuracy :",accuracy)

'''

로스: 0.034642551094293594
R2 score 0.9536113794489123
accuracy : 0.9824561476707458

로스: 0.0669967532157898
R2 score 0.9115674319991807
accuracy : 0.9649122953414917
'''
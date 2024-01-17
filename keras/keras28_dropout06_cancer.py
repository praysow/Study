import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense,Dropout
from sklearn.metrics import accuracy_score
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


x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8, random_state=450)

x_train=np.asarray(x_train).astype(np.float32)
x_test=np.asarray(x_test).astype(np.float32)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
scaler = MaxAbsScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2.모델구성
model=Sequential()
model.add(Dense(10,input_dim=30))
model.add(Dropout(0.1))
model.add(Dense(100))
model.add(Dense(110))
model.add(Dense(120))
model.add(Dense(130))
model.add(Dense(1, activation='sigmoid'))


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
model.compile(loss='binary_crossentropy',optimizer='adam',metrics='accuracy')
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
로스: 0.022365087643265724
R2 score 0.9731414984792953
accuracy : 0.9912280440330505

로스: 0.03639177232980728
R2 score 0.9599780516645485
accuracy : 0.9912280440330505   아래하나

'''
#R2 0.62이상
from sklearn.datasets import load_diabetes
import time
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
from keras.layers import Dense,LSTM
import numpy as np
from keras.utils import to_categorical

x_train,x_test,y_train,y_test= train_test_split(x,y,train_size=0.9, random_state=10)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
scaler = MaxAbsScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)



x_train=x_train.reshape(x_train.shape[0],5,2)
x_test=x_test.reshape(x_test.shape[0],5,2)


#2.모델구성
model=Sequential()
model.add(LSTM(32,input_shape=(5,2),activation='relu'))
model.add(Dense(10))
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(1))
model.add(Dense(10))
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(1))

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
model.compile(loss='mse',optimizer='adam',metrics='accuracy')
start_time = time.time()
hist= model.fit(x_train, y_train, epochs=1000,batch_size=1000, validation_split=0.1,verbose=2,
          callbacks=[es,mcp])
end_time = time.time()

model.save("c:\_data\_save\diabets_1.h5")

#4.결과예측
loss=model.evaluate(x_test,y_test)
y_predict=model.predict([x_test])
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score
# r2=r2_score(y_test,y_predict)
print("로스 ,",loss)
# print("R2 score :",r2)
print("걸린시간 :",round(end_time - start_time))


'''
로스 , 2246.01953125
R2 score : 0.6280751199789454

로스 , 2249.0537109375
R2 score : 0.6275726473646075       위에 하나

로스 , [2166.91259765625, 0.0]
'''
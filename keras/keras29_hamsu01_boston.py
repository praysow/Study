from sklearn.datasets import load_boston
import numpy as np
datasets = load_boston()
x = datasets.data
y = datasets.target

# print(x.shape,y.shape)      (506, 13) (506,)


from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input
import numpy as np

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.9,random_state=100)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
#x_train = scaler.fit_transform(x_train)



# print(np.min(x_train))
# print(np.min(x_test))
# print(np.max(x_train))
# print(np.max(x_test))


# model=Sequential()
# model.add(Dense(1,input_dim=13))
# model.add(Dropout(0.1))
# model.add(Dense(100))
# model.add(Dropout(0.1))                     
# model.add(Dense(1))
# model.add(Dropout(0.1))                    
# model.add(Dense(100))
# model.add(Dropout(0.1))
# model.add(Dense(1))
# model.add(Dense(100))
# model.add(Dense(1))
# model.add(Dense(100))
# model.add(Dense(1))

input1= Input(shape=(13,))
dense1= Dense(1)(input1)
drop1=Dropout(0.1)(dense1)
dense2= Dense(100)(dense1)
drop1=Dropout(0.1)(dense2)
dense3= Dense(1)(dense2)
drop1=Dropout(0.1)(dense3)
dense4= Dense(100)(dense3)
drop1=Dropout(0.1)(dense4)
dense5= Dense(1)(dense4)
dense6= Dense(100)(dense5)
dense7= Dense(1)(dense6)
dense8= Dense(100)(dense7)
output= Dense(1)(dense8)
model = Model(inputs=input1,outputs=output)




#3.컴파일 훈련
from keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime
date= datetime.datetime.now()
print(date)     #10:52:52.279279
date = date.strftime("%m-%d_%H-%M")
print(date) 
print(type(date)) 

path='..//_data//_save//MCP/k27/'
filename= "{epoch:04d}-{val_loss:.4f}.hdf5"  
filepath = "".join([path,'01_boston_',date,'_',filename])

mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    verbose=1,
    save_best_only=True,
    filepath=filepath
    )
es= EarlyStopping(monitor='val_loss',mode='min',patience=50,verbose=1,restore_best_weights=True)
model.compile(loss='mse',optimizer='adam')
hist= model.fit(x_train, y_train, epochs=1000,batch_size=1000, validation_split=0.1,verbose=2,
          callbacks=[es,mcp])

model.save("c:\_data\_save\\boston1.h5")

#4.결과예측
loss=model.evaluate(x_test,y_test)
y_predict=model.predict([x_test])
result=model.predict(x)
print("로스 :",loss)
# print("x 예측값",result)

import matplotlib.pyplot as plt

from sklearn.metrics import r2_score
r2=r2_score(y_test,y_predict)
print("R2 score",r2)

'''
로스 : 44.932708740234375       
R2 score 0.4322480405148875

로스 : 16.97618865966797
R2 score 0.7854955841351186     아래 네줄뺌

로스 : 16.026077270507812
R2 score 0.7975008023366269
'''
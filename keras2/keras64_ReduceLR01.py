#keras25_5copy
from sklearn.datasets import load_boston,load_breast_cancer
from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model
from keras.layers import Dense
from sklearn.metrics import r2_score
import numpy as np

datasets = load_breast_cancer()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.9,random_state=100,stratify=y)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# model= load_model('..\_data\_save\MCP\keras25_MCP1.hdf5')


model=Sequential()
model.add(Dense(1,input_shape=(30,)))
model.add(Dense(100))
model.add(Dense(1))
model.add(Dense(100))
model.add(Dense(1))
model.add(Dense(100))
model.add(Dense(1))
model.add(Dense(100))
model.add(Dense(1))

#3.컴파일 훈련
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
import datetime
date= datetime.datetime.now()
date = date.strftime("%m-%d_%H-%M")

lr = 0.01

path='c://_data//_save//MCP/'
filename= "{epoch:04d}-{val_loss:.4f}.hdf5"                #epoch:04d는 4자리 숫자까지 표현 val_loss:.4f는 소수4자리까지 표현
filepath = "".join([path,'k25_',date,'_'])            #join의 의미 :filepath라는 공간을 만들어 path,date,filename을 서로 연결해 주세요

es= EarlyStopping(monitor='val_loss',mode='min',patience=10,verbose=1,restore_best_weights=False)
mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    verbose=1,
    save_best_only=True,
    filepath=filepath
    )
rlr = ReduceLROnPlateau(monitor='val_loss',patience=10,mode='auto',verbose=1,factor=0.5)

model.compile(loss='binary_crossentropy',optimizer=Adam(learning_rate=lr))
hist= model.fit(x_train, y_train, epochs=100,batch_size=1000, validation_split=0.1,verbose=2, callbacks=[es,mcp,rlr])


#4.결과예측
loss=model.evaluate(x_test,y_test,verbose=0)
y_predict=model.predict(x_test,verbose=0)
result=model.predict(x)
r2=r2_score(y_test,y_predict)
print("R2 score",r2)
print("로스 :",loss)
print("lr:{0},acc:{1}".format(lr,r2))


#restore_best_weights
#save_best_only
#True, True     R2 score 0.010760000953027427
                #로스 : 78.28987884521484
#True, FalseR2 score 0.19254995566007704
#              로스 : 63.90275573730469
#False, TrueR2 score -0.3303490463183649
               #로스 : 105.28572845458984
#False, False   R2 score -0.8165784608691153
                #로스 : 143.7666473388672


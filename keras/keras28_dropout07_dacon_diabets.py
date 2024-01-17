from sklearn.metrics import r2_score, mean_squared_error, mean_squared_log_error
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense,Dropout
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

#1. 데이터

path= "c:\_data\dacon\cancer\\"
train_csv = pd.read_csv(path+"train.csv",index_col=0)
test_csv = pd.read_csv(path+"test.csv",index_col=0)
sampleSubmission_csv = pd.read_csv(path+"sample_Submission.csv")

print("train",train_csv.shape)      #(652,9)
print("test",test_csv.shape)       #(116, 8)
print("sub",sampleSubmission_csv.shape) #(116,2)

x= train_csv.drop(['Outcome'], axis=1)
y= train_csv['Outcome']
x_train,x_test,y_train,y_test=train_test_split(x,y, train_size=0.9, random_state=8)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
scaler = MaxAbsScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2.모델구성
model=Sequential()
model.add(Dense(10,input_dim=8))
model.add(Dense(40))
model.add(Dense(70))
model.add(Dense(100))
model.add(Dense(130))
model.add(Dense(200))
model.add(Dense(130))
model.add(Dense(100))
model.add(Dense(70))
model.add(Dense(40))
model.add(Dense(10))
model.add(Dense(1,activation='sigmoid'))


# 3.컴파일 훈련
from keras.callbacks import EarlyStopping, ModelCheckpoint


import datetime
date= datetime.datetime.now()
print(date)     
date = date.strftime("%m-%d_%H-%M")
print(date) 
print(type(date)) 

path='..//_data//_save//MCP/k27/'
filename= "{epoch:04d}-{val_loss:.4f}.hdf5"  
filepath = "".join([path,'07dacon-diabets_',date,'_',filename])

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

model.save("c:\_data\_save\dacon_diabets_1.h5")

#4.결과예측
loss=model.evaluate(x_test,y_test)
y_submit=model.predict(test_csv)
# y_submit=abs(model.predict(test_csv))

sampleSubmission_csv['Outcome'] = np.round(y_submit)
print(sampleSubmission_csv)
sampleSubmission_csv.to_csv(path +"제출_13.csv", index=False)
# print("로스 :",loss)
# print("음수 개수:",sampleSubmission_csv[sampleSubmission_csv['count']<0].count())

loss,accuracy=model.evaluate(x_test,y_test)
y_predcit=model.predict([x_test])
result=model.predict(x)

def ACC(x_train,y_train):
    return accuracy_score(y_test,np.round(y_predcit))
acc = ACC(y_test,y_predcit)
print("ACC :",acc)
print("로스 :",loss)

'''
ACC : 0.7121212121212122
로스 : 0.47388777136802673

ACC : 0.7727272727272727
로스 : 0.4568730294704437   전부있음
'''
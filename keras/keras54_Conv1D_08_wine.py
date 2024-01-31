import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM,Conv1D,Flatten
from keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder

#1.데이터
datasets= load_wine()
x= datasets.data
y= datasets.target
y_ohe1= to_categorical(datasets.target)

print(x.shape)
print(y_ohe1)
# print(np.unique(y,return_counts=True))

# print(y_ohe1)       (178, 3)
# print(y_ohe1.shape)

# print(x.shape,y.shape)      #(178, 13) (178,)
# print(pd.value_counts(y))
#1    71
#0    59
#2    48
# #pandas
y_ohe2 = pd.get_dummies(y)
print(y_ohe2)      # (178, 3)
# # print(y_ohe2.shape)

# # 사이킷런
y=y.reshape(-1,1)

ohe = OneHotEncoder(sparse=False)
ohe = OneHotEncoder()
y_ohe3= ohe.fit_transform(y).toarray()

# print(y_ohe3.shape)     (178, 3)

x=x.reshape(x.shape[0],13,1)

x_train, x_test, y_train, y_test = train_test_split(x, y_ohe3, train_size=0.8,
                                                    random_state=383,        
                                                    # stratify=y_ohe1            
                                                    )

# from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
# scaler = StandardScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)


#2.모델구성
model=Sequential()
model.add(Conv1D(filters=10,kernel_size=2,input_shape=(13,1),activation='relu'))
model.add(Flatten())
model.add(Dense(50))
model.add(Dense(60))
model.add(Dense(70))
model.add(Dense(60))
model.add(Dense(100))
model.add(Dense(3, activation='softmax'))

# input1= Input(shape=(13,))
# dense1= Dense(50)(input1)
# drop1=Dropout(0.1)(dense1)
# dense2= Dense(60)(dense1)
# drop1=Dropout(0.1)(dense2)
# dense3= Dense(70)(dense2)
# drop1=Dropout(0.1)(dense3)
# dense4= Dense(60)(dense3)
# dense5= Dense(100)(dense4)
# output= Dense(3)(dense5)
# model = Model(inputs=input1,outputs=output)

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
filepath = "".join([path,'08wine_',date,'_',filename])

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

model.save("c:\_data\_save\wine_1.h5")
#4.결과예측
result = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)

y_test = np.argmax(y_test,axis=1)
y_predict= np.argmax(y_predict,axis=1)

# print(y_test)
# print(y_predict)
# print(y_test.shape,y_predict.shape)


from sklearn.metrics import accuracy_score
acc = accuracy_score(y_predict,y_test)
print("accuracy_score :", acc)
print("로스 :", result[0])
print("acc :",result[1])

'''
accuracy_score : 0.9444444444444444
로스 : 0.11515446752309799
acc : 0.9444444179534912

accuracy_score : 1.0
로스 : 0.015490112826228142
acc : 1.0                       아래두개

accuracy_score : 0.9444444444444444
로스 : 0.2360297292470932
acc : 0.9444444179534912

accuracy_score : 0.9722222222222222
로스 : 0.060165151953697205
acc : 0.9722222089767456
'''
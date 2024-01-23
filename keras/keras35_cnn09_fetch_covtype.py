import numpy as np
import pandas as pd
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Conv2D, AveragePooling2D, Flatten
from keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder
import time
#1.데이터
datasets= fetch_covtype()
x= datasets.data
y= datasets.target
y_ohe1= to_categorical(datasets.target)
y_ohe1= y_ohe1[:,1:]
# print(y_ohe1.shape) (581012, 8)
# print(np.unique(y,return_counts=True))



# for i in range(8):
#     print(np.unique(y_ohe1[:,i],return_counts=True))               

# print(y_ohe1.shape)

# print(x.shape,y.shape)      # (581012, 54) (581012,)
# print(pd.value_counts(y))
#1    71
#0    59
#2    48
# #pandas
y_ohe2 = pd.get_dummies(y)
# print(y_ohe2.shape)       (581012, 7)
# # print(y_ohe2.shape)


# # 사이킷런
y=y.reshape(-1,1)

ohe = OneHotEncoder(sparse=False)
ohe = OneHotEncoder()
y_ohe3= ohe.fit_transform(y).toarray()
#print(y_ohe3.shape)     (581012, 7)

print(y_ohe3)

x_train, x_test, y_train, y_test = train_test_split(x, y_ohe1, train_size=0.86,
                                                    random_state=5,        #346
                                                    # stratify=y_ohe1            
                                                    )

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
# scaler = MinMaxScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)

x_train = x_train.reshape(x_train.shape[0],18,3,1)
x_test = x_test.reshape(x_test.shape[0],18,3,1)


# (581012, 8)




#2.모델구성
model=Sequential()
model.add(Conv2D(32, (2,2),input_shape=(18,3,1),activation='relu'))
model.add(AveragePooling2D())
model.add(Dropout(0.1))
model.add(Flatten())
model.add(Dropout(0.1))
model.add(Dense(70,activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(80,activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(90,activation='relu'))
model.add(Dense(100,activation='relu'))
model.add(Dense(7, activation='softmax'))

# input1= Input(shape=(54,))
# dense1= Dense(30,activation='relu')(input1)
# drop1=Dropout(0.1)(dense1)
# dense2= Dense(70,activation='relu')(dense1)
# drop1=Dropout(0.1)(dense2)
# dense3= Dense(80,activation='relu')(dense2)
# drop1=Dropout(0.1)(dense3)
# dense4= Dense(90,activation='relu')(dense3)
# dense5= Dense(100,activation='relu')(dense4)
# output= Dense(7, activation='softmax')(dense5)
# model = Model(inputs=input1,outputs=output)

# 3.컴파일 훈련
from keras.callbacks import EarlyStopping, ModelCheckpoint

import datetime
date= datetime.datetime.now()
print(date)     
date = date.strftime("%m-%d_%H-%M")
print(date) 
print(type(date)) 

path='..//_data//_save//MCP/k26/'
filename= "{epoch:04d}-{val_loss:.4f}.hdf5"  
filepath = "".join([path,'09_fetch_covtype_',date,'_',filename])

mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    verbose=1,
    save_best_only=True,
    filepath=filepath
    )
es= EarlyStopping(monitor='val_loss',mode='min',patience=100,verbose=1,restore_best_weights=True)
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics='accuracy')
start_time = time.time()

hist= model.fit(x_train, y_train, epochs=100,batch_size=100000, validation_split=0.1,verbose=2,
          callbacks=[es,mcp])
end_time = time.time()

model.save("c:\_data\_save\\fetch_covtype_1.h5")

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

print("걸린시간 :",round(end_time - start_time))


'''
accuracy_score : 0.841262816257284
로스 : 0.3754711449146271
acc : 0.8412628173828125
random값 : 892

accuracy_score : 0.7687418553760664
로스 : 0.5514593124389648
acc : 0.7687418460845947

accuracy_score : 0.5281896191389442
로스 : 1.0075305700302124
acc : 0.5281895995140076
걸린시간 : 29
'''
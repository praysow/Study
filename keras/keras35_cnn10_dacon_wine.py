from keras.models import Sequential
from keras.layers import Dense, Dropout, AveragePooling2D, Flatten, Conv2D
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd

#1. 데이터
path= "c:\_data\dacon\wine\\"
train_csv = pd.read_csv(path+"train.csv",index_col=0)
test_csv = pd.read_csv(path+"test.csv",index_col=0)
sampleSubmission_csv = pd.read_csv(path+"sample_Submission.csv")
x= train_csv.drop(['quality'], axis=1)
y= train_csv['quality']
# x['type'] = x['type'].map({'red': 1, 'white': 2})
# y['type'] = y['type'].map({'red': 1, 'white': 2})

# print(train_csv.shape)  (5497, 13)
# print(test_csv.shape)   (1000, 12)

y_ohe= pd.get_dummies(y)

# print(y_ohe)
# print(y_ohe.shape)      (5497, 7)

# print(np.unique(y,return_counts=True))
# (array([3, 4, 5, 6, 7, 8, 9], dtype=int64), array([  26,  186, 1788, 2416,  924,  152,    5], dtype=int64))
# x_train,x_test,y_train,y_test= train_test_split(x,y,)
# print(train_csv.isna().sum()) 
# print(test_csv.isna().sum())
lb=LabelEncoder()
lb.fit(x['type'])
x['type'] =lb.transform(x['type'])
test_csv['type'] =lb.transform(test_csv['type'])

x_train,x_test,y_train,y_test=train_test_split(x,y_ohe, train_size= 0.9193904973982694, random_state=1909,
                                            stratify=y_ohe)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)



x_train=np.asarray(x_train).astype(np.float32)
x_test=np.asarray(x_test).astype(np.float32)
test_csv = np.asarray(test_csv).astype(np.float32)

x_train=x_train.reshape(x_train.shape[0],3,2,2)
x_test=x_test.reshape(x_test.shape[0],3,2,2)
test_csv=test_csv.reshape(test_csv.shape[0],3,2,2)



#2.모델구성
model=Sequential()
model.add(Conv2D(10, (1,1),input_shape=(3,2,2),activation='relu'))
model.add(AveragePooling2D())
model.add(Flatten())
model.add(Dense(139))
model.add(Dense(151))
model.add(Dropout(0.1))
model.add(Dense(97))
model.add(Dense(127))
model.add(Dropout(0.1))
model.add(Dense(57))
model.add(Dense(88))
model.add(Dense(164))
model.add(Dense(151))
model.add(Dropout(0.1))
model.add(Dense(154,activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(7,activation='softmax'))


#3. 컴파일 훈련
from keras.callbacks import EarlyStopping, ModelCheckpoint

import datetime
date= datetime.datetime.now()
print(date)     
date = date.strftime("%m-%d_%H-%M")
print(date) 
print(type(date)) 

path='..//_data//_save//MCP/k26/'
filename= "{epoch:04d}-{val_loss:.4f}.hdf5"  
filepath = "".join([path,'10_dacon_wine_',date,'_',filename])

mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    verbose=1,
    save_best_only=True,
    filepath=filepath
    )
es= EarlyStopping(monitor='val_loss',mode='min',patience=100,verbose=1,restore_best_weights=True)
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics='accuracy')
hist= model.fit(x_train, y_train, epochs=10,batch_size=1000, validation_split=0.1,verbose=2,
          callbacks=[es,mcp])

model.save("c:\_data\_save\dacon_wine_1.h5")


#4.결과예측
loss= model.evaluate(x_test,y_test)
y_submit= model.predict(test_csv)
y_test= np.argmax(y_test,axis=1)
y_submit = np.argmax(y_submit,axis=1)+3

sampleSubmission_csv['quality'] = (y_submit)
# print(sampleSubmission_csv)
sampleSubmission_csv.to_csv(path + "submission_9.csv", index=False)

y_predict=model.predict(x_test)
y_predict= np.argmax(y_predict,axis=1)          #+3을 한 이유는 y에 유니크를 찍었을때는 3,4,5,6,7,8,9로 나왔는데 
                                                    #y_predict는 0,1,2,3,4,5,6,7 이런식으로 나왔기 때문에 데이콘에 제출을 위해서 뒤로 3칸씩 미는 작업을 한것이다




def ACC(x_train,y_train):
    return accuracy_score(y_test,np.round(y_predict))
acc = ACC(y_test,y_predict)
print("ACC :",acc)
print("로스 :",loss)
'''
ACC : 0.5833333333333334
로스 : [1.0416243076324463, 0.5833333134651184]         load


ACC : 0.6216216216216216
로스 : [0.9659953117370605, 0.6216216087341309]     아래하나
'''
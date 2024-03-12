import numpy as np
import pandas as pd
from keras.models import Sequential, load_model
from keras.layers import Dense,Dropout,BatchNormalization
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler, Normalizer, RobustScaler
from sklearn.metrics import accuracy_score, f1_score
from keras.utils import to_categorical
#1.데이터
path= "c:\_data\dacon\dechul\\"
train_csv=pd.read_csv(path+"train.csv",index_col=0)
test_csv=pd.read_csv(path+"test.csv",index_col=0)
sample_csv=pd.read_csv(path+"sample_submission.csv")
x= train_csv.drop(['대출등급'],axis=1)
y= train_csv['대출등급']


# print(train_csv,train_csv.shape)        (96294, 14)
# print(test_csv,test_csv.shape)          (64197, 13)
# print(sample_csv,sample_csv.shape)      (64197, 2)
print(np.unique(y,return_counts=True))



y=y.values.reshape(-1,1)

ohe = OneHotEncoder(sparse=False)
ohe = OneHotEncoder()
y_ohe = ohe.fit_transform(y).toarray()

# print(y_ohe,y_ohe.shape)


lb=LabelEncoder()
lb.fit(x['대출기간'])
x['대출기간'] = lb.transform(x['대출기간'])
lb.fit(x['근로기간'])
x['근로기간'] = lb.transform(x['근로기간'])
lb.fit(x['주택소유상태'])
x['주택소유상태'] = lb.transform(x['주택소유상태'])
lb.fit(x['대출목적'])
x['대출목적'] = lb.transform(x['대출목적'])

lb.fit(test_csv['대출기간'])
test_csv['대출기간'] =lb.transform(test_csv['대출기간'])

lb.fit(test_csv['근로기간'])
test_csv['근로기간'] =lb.transform(test_csv['근로기간'])

lb.fit(test_csv['주택소유상태'])
test_csv['주택소유상태'] =lb.transform(test_csv['주택소유상태'])

lb.fit(test_csv['대출목적'])
test_csv['대출목적'] =lb.transform(test_csv['대출목적'])


x_train,x_test,y_train,y_test=train_test_split(x,y_ohe,train_size=0.9,random_state=3 ,
                                               stratify=y_ohe
                                               )

scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)


# scaler = StandardScaler()
# x_train_scaled = scaler.fit_transform(x_train)
# x_test_scaled = scaler.transform(x_test)
# test_csv_scaled = scaler.transform(test_csv)
# print(x_train,x_test)
# print(y_train,y_test)

#2.모델구성
model=Sequential()
model.add(Dense(7,input_shape=(13,),activation='relu'))
model.add(Dense(44,activation='relu'))
model.add(Dense(40,activation='relu'))
model.add(Dense(34,activation='relu'))
model.add(Dense(26,activation='relu'))
model.add(BatchNormalization(axis=1))
model.add(Dense(20,activation='relu'))
model.add(Dense(30,activation='relu'))
model.add(Dense(7,activation='softmax'))

# model= load_model("c:\_data\_save\대출모델9.h5")
#3.컴파일 훈련
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
import datetime
date= datetime.datetime.now()
date = date.strftime("%m-%d_%H-%M")

lr = 0.1

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
hist= model.fit(x_train, y_train, epochs=100,batch_size=3000, validation_split=0.1,verbose=2, callbacks=[es,mcp,rlr])

# 4.결과예측
loss = model.evaluate(x_test, y_test)
y_submit = model.predict(test_csv)
y_test_indices = np.argmax(y_test, axis=1)
y_submit_indices = np.argmax(y_submit, axis=1)

# 할당 전에 길이 확인
# print(len(y_test_indices), len(y_submit_indices), len(sample_csv))

# y_test_indices = np.argmax(y_test, axis=1)                        (argmax는 숫자가 제일 큰곳의 인덱스를 알려줌)
# y_submit_indices = np.argmax(y_submit, axis=1)

y_submit = ohe.inverse_transform(y_submit)
y_submit = pd.DataFrame(y_submit)
sample_csv["대출등급"]=y_submit

sample_csv.to_csv(path + "sample_submission_19.csv", index=False)

y_pred= model.predict(x_test)
y_pred= ohe.inverse_transform(y_pred)
y_test = ohe.inverse_transform(y_test)
f1=f1_score(y_test,y_pred, average='macro')
# y_pred_binary = (y_pred > 0.5).astype(float)
acc = accuracy_score(y_test, y_pred)
print("f1",f1)
print("lr:{0},acc:{1}".format(lr,acc))
print("로스 :",loss)

'''
lr:0.1,acc:0.29968847352024924
로스 : 0.3532910645008087
lr:0.01,acc:0.5021806853582554
로스 : 0.37821483612060547
lr:0.001,acc:0.28598130841121494
로스 : [1.9059311151504517, 0.28598129749298096]
lr:0.0001,acc:0.29989615784008306
로스 : [2.0212290287017822, 0.29989615082740784]
'''
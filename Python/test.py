import numpy as np
import pandas as pd
from keras.models import Sequential, load_model ,Model
from keras.layers import Dense,Dropout,BatchNormalization, Input,Conv1D,MaxPooling1D,Flatten
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler, Normalizer, RobustScaler
from sklearn.metrics import accuracy_score, f1_score
import time
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
# print(np.unique(y,return_counts=True))



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


x_train,x_test,y_train,y_test=train_test_split(x,y_ohe,train_size=0.9,random_state=333 ,stratify=y_ohe)

scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)


# print(x_train,x_test)
# print(y_train,y_test)


from keras import regularizers
r1=int(np.random.uniform(1,100))
r2=int(np.random.uniform(1,100))
r3=int(np.random.uniform(1,100))
r4=int(np.random.uniform(1,100))
r5=int(np.random.uniform(1,100))
r6=int(np.random.uniform(1,100))
r7=int(np.random.uniform(1,100))

#2.모델구성
model=Sequential()
model.add(Dense(r1,input_shape=(13,),activation='relu'))
model.add(Dense(r2,activation='relu'))
model.add(Dense(r3,activation='relu'))
model.add(Dense(r4,activation='relu',))
model.add(Dense(r5, activation='relu'))
model.add(BatchNormalization(axis=1))
model.add(Dropout(0.1))
model.add(Dense(r6,activation='relu'))
model.add(Dense(r7,activation='relu', kernel_regularizer=regularizers.l2(0.1)))
model.add(Dense(7,activation='softmax'))

# model= load_model("c:\_data\_save\대출모델9.h5")
#3.컴파일 훈련
# input1= Input(shape=(13,))
# dense1= Dense(8,activation='relu')(input1)
# dense2= Dense(44,activation='relu')(input1)
# dense3= Dense(44,activation='relu')(dense2)
# dense4= Dense(30,activation='relu')(dense3)
# dense5= Dense(24,activation='relu')(dense4)
# drop1=BatchNormalization(axis=1)(dense5)
# dense6= Dense(2,activation='relu')(drop1)
# dense7= Dense(48,activation='relu')(dense6)
# output= Dense(7,activation='softmax')(dense7)
# model = Model(inputs=input1,outputs=output)


from keras.callbacks import EarlyStopping,ModelCheckpoint
es= EarlyStopping(monitor='val_loss',mode='auto',patience=1000,verbose=1,restore_best_weights=True)
mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    verbose=1,
    save_best_only=True,
    filepath='..\_data\_save\MCP\대출_31.hdf5'
    )

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
start_time = time.time()

hist= model.fit(x_train, y_train, epochs=100000,batch_size=2000, validation_split=0.1,verbose=3,
          callbacks=[es,mcp]
            )
# model=load_model("c:\_data\_save\dechul_8.h5")
end_time = time.time()

model.save("c:\_data\_save\대출_31.h5")


# ... (이전 코드)

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

sample_csv.to_csv(path + "대출_31.csv", index=False)

y_pred= model.predict(x_test)
y_pred= ohe.inverse_transform(y_pred)
y_test = ohe.inverse_transform(y_test)
f1=f1_score(y_test,y_pred, average='macro')


print("f1",f1)
print("로스:", loss[0])
print("acc", loss[1])
print("걸린시간 :",round(end_time - start_time))

print("r1",r1)
print("r2",r2)
print("r3",r3)
print("r4",r4)
print("r5",r5)
print("r6",r6)
print("r7",r7)
'''
f1 0.8002436934022501         gpu
로스: 0.46971189975738525
acc 0.8276219964027405
걸린시간 : 72

f1 0.9425043684945906       28번
로스: 0.15422487258911133
acc 0.9418483972549438
걸린시간 : 1440

f1 0.9196787225477553
로스: 0.1654994785785675
acc 0.9409137964248657
걸린시간 : 1314

f1 0.9206822434519886
로스: 0.18335647881031036   29번
acc 0.9366562962532043
걸린시간 : 1350

f1 0.927115626970566      30번
로스: 0.17489323019981384
acc 0.9393562078475952
걸린시간 : 1772

f1 0.920310034691761        31번
로스: 0.18700696527957916
acc 0.9352024793624878
걸린시간 : 857
r1 68
r2 7
r3 41
r4 15
r5 65
r6 52
r7 79
 '''

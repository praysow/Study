import numpy as np
import pandas as pd
from keras.models import Sequential, load_model
from keras.layers import Dense,Dropout,BatchNormalization, AveragePooling2D, Flatten, Conv2D, LSTM, Bidirectional,Conv1D
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler, Normalizer, RobustScaler
from sklearn.metrics import accuracy_score, f1_score
from keras.utils import to_categorical
#1.데이터
path= "c:\_data\dacon\dechul\\"
train_csv=pd.read_csv(path+"train.csv",index_col=0)
test_csv=pd.read_csv(path+"test.csv",index_col=0)
sample_csv=pd.read_csv(path+"sample_submission.csv")
# train_csv = train_csv[train_csv['총상환이자'] != 0.0]
x= train_csv.drop(['대출등급','최근_2년간_연체_횟수','총연체금액','연체계좌수'],axis=1)
y= train_csv['대출등급']
test_csv = test_csv.drop(['최근_2년간_연체_횟수','총연체금액','연체계좌수'],axis=1)

# print(x.shape)
# print(x.shape)      #  (96294, 14)
# print(test_csv.shape)       #   (64197, 13)
# print(sample_csv.shape)  #    (64197, 2)
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


x_train,x_test,y_train,y_test=train_test_split(x,y_ohe,train_size=0.85,random_state=3 ,
                                               stratify=y_ohe
                                               )

scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)

# x_train=x_train.reshape(x_train.shape[0],10,1)
# x_test=x_test.reshape(x_test.shape[0],10,1)
# test_csv=test_csv.reshape(test_csv.shape[0],10,1)

# print(x_train,x_test)
# print(y_train,y_test)

#2.모델구성
# model=Sequential()
# model.add(Bidirectional(LSTM(7),input_shape=(13,1)))
# model.add(Dense(44,activation='relu'))
# model.add(Dense(44,activation='relu'))
# model.add(Dense(30,activation='relu'))
# model.add(Dense(24,activation='relu'))
# model.add(Dense(2,activation='relu'))
# model.add(Dense(48,activation='relu'))
# model.add(Dense(7,activation='softmax'))
#swish
# model= load_model("c:\_data\_save\대출모델9.h5")
model=Sequential()
# model.add(Bidirectional(LSTM(7),input_shape=(13,1)))
# model.add(Conv1D(filters=7,kernel_size=2,input_shape=(10,1)))
model.add(Dense(10,input_shape=(10,)))
# model.add(Conv1D(filters=15,kernel_size=2,activation='swish'))
# model.add(Flatten())
# model.add(Dense(15,activation='swish'))
# model.add(Dense(21,activation='swish'))
# model.add(Dense(23,activation='swish'))
# model.add(Dense(8,activation='swish'))
# model.add(Dense(17,activation='swish'))
# model.add(Dense(14,activation='swish'))
# model.add(Dense(14,activation='swish'))
# model.add(Dense(5,activation='swish'))
# model.add(Dense(230,activation='swish'))
# model.add(Dense(11,activation='swish'))
# model.add(Dense(14,activation='swish'))
# model.add(Dense(20,activation='swish'))
# model.add(Dense(12,activation='swish'))
# model.add(Dense(15,activation='swish'))
# model.add(Dense(21,activation='swish'))
# model.add(Dense(230,activation='swish'))
# model.add(Dense(8,activation='swish'))
# model.add(Dense(17,activation='swish'))
# model.add(Dense(14,activation='swish'))
# model.add(Dense(14,activation='swish'))
# model.add(Dense(5,activation='swish'))
# model.add(Dense(230,activation='swish'))
# model.add(BatchNormalization())
# model.add(Dense(11,activation='swish'))
# model.add(Dense(14,activation='swish'))
# model.add(Dense(20,activation='swish'))
# model.add(Dense(12,activation='swish'))
# model.add(Dense(7,activation='softmax'))
model.add(Dense(58,activation='swish'))
model.add(Dense(82,activation='swish'))
model.add(Dense(85,activation='swish'))
model.add(Dense(85,activation='swish'))
model.add(Dense(77,activation='swish'))
model.add(Dense(69,activation='swish'))
model.add(Dense(74,activation='swish'))
model.add(Dense(58,activation='swish'))
model.add(Dense(82,activation='swish'))
model.add(Dense(85,activation='swish'))
model.add(Dense(85,activation='swish'))
model.add(Dense(77,activation='swish'))
model.add(Dense(69,activation='swish'))
model.add(Dense(74,activation='swish'))
model.add(Dense(58,activation='swish'))
model.add(Dense(82,activation='swish'))
model.add(Dense(85,activation='swish'))
model.add(Dense(85,activation='swish'))
model.add(BatchNormalization())
model.add(Dense(77,activation='swish'))
model.add(Dense(69,activation='swish'))
model.add(Dense(74,activation='swish'))
model.add(Dense(7,activation='softmax'))

#3.컴파일 훈련

# model.summary()
from keras.callbacks import EarlyStopping,ModelCheckpoint
es= EarlyStopping(monitor='val_loss',mode='min',patience=2000,verbose=1,restore_best_weights=True)
mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    verbose=1,
    save_best_only=True,
    filepath='..\_data\_save\MCP\대출81.hdf5'
    )

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
hist= model.fit(x_train, y_train, epochs=100000,batch_size=3000, validation_split=0.1,verbose=2,
          callbacks=[es,mcp]
            )
# model=load_model("c:\_data\_save\dechul_8.h5")
model.save("c:\_data\_save\대출81.h5")


# ... (이전 코드)

# 4.결과예측
loss = model.evaluate(x_test, y_test)
y_submit = model.predict(test_csv)
y_test_indices = np.argmax(y_test, axis=1)
y_submit_indices = np.argmax(y_submit, axis=1)

# 할당 전에 길이 확인
# print(len(y_test_indices), len(y_submit_indices), len(sample_csv))


y_submit = ohe.inverse_transform(y_submit)
y_submit = pd.DataFrame(y_submit)
sample_csv["대출등급"]=y_submit

sample_csv.to_csv(path + "대출81.csv", index=False)

y_pred= model.predict(x_test)
y_pred= ohe.inverse_transform(y_pred)
y_test = ohe.inverse_transform(y_test)
f1=f1_score(y_test,y_pred, average='macro')


print("f1",f1)
print("로스:", loss[0])
print("acc", loss[1])

'''
f1 0.9555395033102171       80번
로스: 0.09324903786182404
acc 0.9742982983589172

'''

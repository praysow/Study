import numpy as np
import pandas as pd
from keras.models import Sequential, load_model
from keras.layers import Dense,Dropout,BatchNormalization
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler
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


x_train,x_test,y_train,y_test=train_test_split(x,y_ohe,train_size=0.9,random_state=3,
                                               stratify=y_ohe
                                               )

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)

#2.모델구성
# r1=int(np.random.uniform(1,50))
# r2=int(np.random.uniform(1,50))
# r3=int(np.random.uniform(1,50))
# r4=int(np.random.uniform(1,50))
# r5=int(np.random.uniform(1,50))
# r6=int(np.random.uniform(1,50))
# r0=int(np.random.uniform(1,1000))


# #2.모델구성
# model=Sequential()
# model.add(Dense(r1,input_shape=(13,),activation='relu'))
# model.add(Dense(r2,activation='relu'))
# model.add(Dense(r2,activation='relu'))
# model.add(Dense(r3,activation='relu'))
# model.add(Dense(r4,activation='relu'))
# model.add(Dense(r5,activation='relu'))
# model.add(Dense(r6,activation='relu'))
# model.add(Dense(7,activation='softmax'))


# #3.컴파일 훈련

# from keras.callbacks import EarlyStopping,ModelCheckpoint
# es= EarlyStopping(monitor='val_loss',mode='min',patience=1000,verbose=1,restore_best_weights=True)
# mcp = ModelCheckpoint(
#     monitor='val_loss',
#     mode='auto',
#     verbose=1,
#     save_best_only=True,
#     filepath='..\_data\_save\MCP\keras25_MCP1.hdf5'
#     )

# model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
# hist= model.fit(x_train, y_train, epochs=15000,batch_size=3000, validation_split=0.3,verbose=3,
#           callbacks=[es,mcp]
#             )

# model.save("c:\_data\_save\dechul_1.h5")


# ... (이전 코드)

model = load_model("c:\_data\_save\MCP\k26\\dacon_dechul_01-17_15-16_0100-0.4638.hdf5")

# model.summary()



#4.결과예측
loss = model.evaluate(x_test, y_test)
y_submit = model.predict(test_csv)

# 할당 전에 길이 확인
# print(len(y_test_indices), len(y_submit_indices), len(sample_csv))

# y_test_indices = np.argmax(y_test, axis=1)                        (argmax는 숫자가 제일 큰곳의 인덱스를 알려줌)
# y_submit_indices = np.argmax(y_submit, axis=1)

y_submit = ohe.inverse_transform(y_submit)
y_submit = pd.DataFrame(y_submit)
sample_csv["대출등급"]=y_submit

sample_csv.to_csv(path + "sample_submission_5.csv", index=False)

y_pred= model.predict(x_test)
y_pred= ohe.inverse_transform(y_pred)
y_test = ohe.inverse_transform(y_test)
f1=f1_score(y_test,y_pred, average='macro')


print("f1",f1)
print("로스:", loss[0])
print("acc", loss[1])

'''
f1 0.7961885559394487
로스: 0.4505821764469147M
acc 0.845171332359314       save

f1 0.7961885559394487       load
로스: 0.4505821764469147
acc 0.845171332359314
'''
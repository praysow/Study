import numpy as np
import pandas as pd
from keras.models import Sequential
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

train_csv = train_csv.dropna()
#train_csv=train_csv.fillna(train_csv.mean())                         #test는 dropna를 하면 안되고 결측치를 변경해줘야한다
# train_csv=train_csv.fillna(0)
test_csv=test_csv.fillna(test_csv.mean())                         #test는 dropna를 하면 안되고 결측치를 변경해줘야한다
#test_csv=test_csv.fillna(0)


x_train,x_test,y_train,y_test=train_test_split(x,y_ohe,train_size=0.99,random_state=8,
                                               stratify=y_ohe
                                               )

scaler = MinMaxScaler()
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

# x_train=np.asarray(x_train).astype(np.float32)
# x_test=np.asarray(x_test).astype(np.float32)
# test_csv = np.asarray(test_csv).astype(np.float32)


# #2.모델구성
model=Sequential()
model.add(Dense(60,input_shape=(13,),activation='relu'))
model.add(Dense(50))
model.add(Dense(40))
model.add(Dense(30))
model.add(Dense(20))
model.add(Dense(10,activation='relu'))
model.add(Dense(7,activation='softmax'))
# model = Sequential()
# model.add(Dense(28, input_shape=(13,), activation='relu'))
# model.add(BatchNormalization())
# model.add(Dropout(0.5))

# model.add(Dense(56, activation='relu'))
# model.add(BatchNormalization())
# model.add(Dropout(0.5))

# model.add(Dense(56, activation='relu'))
# model.add(BatchNormalization())
# model.add(Dropout(0.5))

# model.add(Dense(28, activation='relu'))
# model.add(BatchNormalization())
# model.add(Dropout(0.3))

# model.add(Dense(7,activation='softmax'))

#3.컴파일 훈련

from keras.callbacks import EarlyStopping
es= EarlyStopping(monitor='val_accuracy',mode='max',patience=1000,verbose=1,restore_best_weights=True)
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
hist= model.fit(x_train, y_train, epochs=10000,batch_size=1000, validation_split=0.1,verbose=2,
          callbacks=[es]
            )




# ... (이전 코드)

#4.결과예측
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

sample_csv.to_csv(path + "sample_submission_1.csv", index=False)

y_pred= model.predict(x_test)
y_pred= ohe.inverse_transform(y_pred)
y_test = ohe.inverse_transform(y_test)
f1=f1_score(y_test,y_pred, average='macro')


print("f1",f1)
print("로스:", loss[0])
print("acc", loss[1])


import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
font_path = "c:\windows\Fonts\gulim.ttc"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)

plt.figure(figsize=(9,6))
plt.plot(hist.history['val_loss'],c='red', label='loss',marker='.')
plt.plot(hist.history['val_accuracy'],c='blue', label='acc',marker='.')
plt.legend(loc='upper right')
plt.title('wine loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.grid()
plt.show()


'''


f1 0.381682505032396
로스: 18.757368087768555
acc 0.4139363467693329

f1 0.46186029602589557             
로스: 368.1031799316406
acc 0.5139415264129639

f1 0.5361996895682537               2번
로스: 2.2290637493133545
acc 0.5927098989486694

f1 0.5195745384263604
로스: 2234.37548828125      3번
acc 0.5906628370285034

f1 0.5507853018857906       4번
로스: 324.82159423828125
acc 0.6089037656784058

f1 0.6737068121973869           csv_.py
로스: 0.5376889109611511        train_size=0.78,random_state=8
acc 0.8047675490379333          epochs=10000,batch_size=3000, validation_split=0.27

f1 0.7090078411917778
로스: 0.43172240257263184
acc 0.8471193909645081
'''
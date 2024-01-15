import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
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

x_train,x_test,y_train,y_test=train_test_split(x,y_ohe,train_size=0.9,random_state=1,stratify=y_ohe)

# print(x_train,x_test)
# print(y_train,y_test)

# x_train=np.asarray(x_train).astype(np.float32)
# x_test=np.asarray(x_test).astype(np.float32)
# test_csv = np.asarray(test_csv).astype(np.float32)

#2.모델구성
model=Sequential()
model.add(Dense(3,input_shape=(13,)))
model.add(Dense(4))
model.add(Dense(5))
model.add(Dense(6))
model.add(Dense(7))
model.add(Dense(7,activation='softmax'))

#3.컴파일 훈련
from keras.callbacks import EarlyStopping
es= EarlyStopping(monitor='val_loss',mode='min',patience=100,verbose=1,restore_best_weights=True)
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10,batch_size=1000, validation_split=0.1,verbose=2,
          callbacks=[es]
            )





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

sample_csv.to_csv(path + "sample_submission_100.csv", index=False)

f1 = f1_score(y_test_indices, y_submit_indices, average='macro')

print("f1",f1)
print("로스:", loss[0])
print("acc", loss[1])


'''
로스1.3  0.32점

'''
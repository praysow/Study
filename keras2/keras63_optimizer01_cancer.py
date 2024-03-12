#keras25_5copy
from sklearn.datasets import load_boston,load_breast_cancer
from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model
from keras.layers import Dense
from sklearn.metrics import r2_score,accuracy_score
import numpy as np

datasets = load_breast_cancer()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.9,random_state=100,stratify=y)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

model=Sequential()
model.add(Dense(1,input_shape=(30,)))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(1,activation='sigmoid'))  # 출력 레이어의 노드 수를 1로 변경하고 활성화 함수를 시그모이드로 설정


#3.컴파일 훈련
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import datetime
date= datetime.datetime.now()
print(date)     #10:52:52.279279
date = date.strftime("%m-%d_%H-%M")
print(date) 
print(type(date))

from keras.optimizers import Adam
learning_rate = 0.1
model.compile(loss='binary_crossentropy',optimizer=Adam(learning_rate=learning_rate))
model.fit(x_train, y_train, epochs=100,batch_size=1000, validation_split=0.1,verbose=2)


#4.결과예측
loss=model.evaluate(x_test,y_test,verbose=0)
y_predict=model.predict(x_test,verbose=0)
result=model.predict(x)
# r2=r2_score(y_test,y_predict)
# print("R2 score",r2)
y_pred = model.predict(x_test)
# y_pred를 이진 형식으로 변환하여 정확도를 계산
y_pred_binary = (y_pred > 0.5).astype(int)
acc = accuracy_score(y_test, y_pred_binary)
print("lr:{0},acc:{1}".format(learning_rate,acc))
print("로스 :",loss)

'''
lr:0.1,acc:0.9473684210526315
로스 : 0.2828892469406128
lr:0.01,acc:0.9473684210526315
로스 : 0.14316314458847046
lr:0.001,acc:0.9649122807017544
로스 : 0.2504443824291229
lr:0.0001,acc:0.8596491228070176
로스 : 0.457588791847229
'''


from sklearn.preprocessing import OneHotEncoder
from keras.datasets import imdb
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense,Embedding,Flatten,SimpleRNN,Conv1D,BatchNormalization

(x_train,y_train),(x_test,y_test) = imdb.load_data(num_words=10000)
# print(y_train[:20])
# print(x_train)
# print(x_train.shape,x_test.shape)   #(25000,) (25000,)
# print(y_train.shape,y_test.shape)   #(25000,) (25000,)
# print(np.unique(y_train,return_counts=True))      #24901
# (array([0, 1], dtype=int64), array([12500, 12500], dtype=int64))
print(len(x_train[0]),len(x_train[1]))#218 189
# print("뉴스 기사의 최대길이 : ",max(len(i) for i in x_train)) #2494
# print("뉴스 기사의 평균길이 : ",sum(map(len, x_train)) / len(x_train))   #238.71364
from keras.utils import pad_sequences
x_train = pad_sequences(x_train,padding = 'pre',maxlen=200, truncating='pre')
x_test = pad_sequences(x_test,padding = 'pre',maxlen=200, truncating='pre')
#원핫사용

# print(x_train.shape,x_test.shape)(8982, 100) (2246, 100)
# print(y_test.shape)

# ohe = OneHotEncoder(sparse=False)
# y_train=y_train.reshape(-1,1)
# y_train = ohe.fit_transform(y_train)
# y_test=y_test.reshape(-1,1)
# y_test = ohe.fit_transform(y_test)

# print(x_train.shape,x_test.shape)#(25000, 200) (25000, 200)
# print(y_test.shape,y_test.shape)#(25000, 2) (25000, 2)
#2.모델구성
model = Sequential()
model.add(Embedding(input_dim=10000,output_dim=40,input_length=200))
model.add(Conv1D(10,2))
model.add(Conv1D(20,2))
model.add(Conv1D(30,2))
model.add(Conv1D(40,2))
model.add(Conv1D(60,2))
model.add(Conv1D(70,2))
model.add(SimpleRNN(60))
model.add(Dense(50))
model.add(Dense(60))
model.add(Dense(30))
model.add(Dense(1,activation='sigmoid'))

#3.컴파일 훈련
from keras.callbacks import EarlyStopping,ModelCheckpoint
es = EarlyStopping(monitor='val_loss',mode='auto',patience=100,verbose=1,restore_best_weights=True)
mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    verbose=1,
    save_best_only=True,
    filepath='..\_data\_save\MCP\imdb.hdf5'
    )
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
hist= model.fit(x_train, y_train, epochs=100,batch_size=3000, validation_split=0.2,verbose=2,
          callbacks=[es,mcp]
            )
#4.결과예측
result=model.evaluate(x_test,y_test)
y_pred=model.predict([x_test])
print("loss:",result[0])
print("acc:",result[1])
print("예측값",y_pred)
'''
loss: 1.891600489616394
acc: 0.797760009765625
'''
from sklearn.preprocessing import OneHotEncoder
from keras.datasets import reuters
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense,Embedding,Flatten,SimpleRNN,Conv1D,BatchNormalization
(x_train,y_train),(x_test,y_test) = reuters.load_data(num_words=230,
                                                      test_split=0.2)
# word_index = reuters.get_word_index()
# vocab_size = len(word_index)
# print("단어 사전의 크기:", vocab_size)   input_dim=30979+1

# print(x_train)
# print(y_train.shape,y_test.shape)  # (8982,) (2246,)
# print(type(x_train))
# print(len(np.unique(x_train))) #x_train=7185,y_train=46
# print(type(x_train[7185]))    #list
# print(len(x_train[0]),len(x_train[1]))  #87 56
#최종 노드의 갯수는 46
# print("뉴스 기사의 최대길이 : ",max(len(i) for i in x_train)) #2376
# print("뉴스 기사의 평균길이 : ",sum(map(len, x_train)) / len(x_train))  # 145.53985
#첫번째 인베딩레이어
#전처리
from keras.utils import pad_sequences
x_train = pad_sequences(x_train,padding = 'pre',maxlen=100, truncating='pre')
x_test = pad_sequences(x_test,padding = 'pre',maxlen=100, truncating='pre')
#원핫사용

# print(x_train.shape,x_test.shape)(8982, 100) (2246, 100)
print(y_test.shape)

ohe = OneHotEncoder(sparse=False)
y_train=y_train.reshape(-1,1)
y_train = ohe.fit_transform(y_train)
y_test=y_test.reshape(-1,1)
y_test = ohe.fit_transform(y_test)
# y_train = ohe.fit_transform(y_train.reshape(-1, 1)).argmax(axis=1)
# y_test = ohe.fit_transform(y_test.reshape(-1, 1)).argmax(axis=1)


# print(y_train,y_test)
# print(x_train.shape,x_test.shape)#(8982, 100) (2246, 100)
# print(y_train.shape,y_test.shape)#(8982,46) (2246,46)


model= Sequential()
model.add(Embedding(input_dim=1000,output_dim=30
                    ,input_length=100
                    ))#input_dim=단어 사전의 갯수 output_dim=훈련의 갯수(노드수),input_length=shape의 갯수
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
model.add(Dense(46,activation='softmax'))
# model.summary()
from keras.callbacks import EarlyStopping,ModelCheckpoint
es= EarlyStopping(monitor='val_loss',mode='min',patience=1000,verbose=1,restore_best_weights=True)
mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    verbose=1,
    save_best_only=True,
    filepath='..\_data\_save\MCP\로이터.hdf5'
    )

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
hist= model.fit(x_train, y_train, epochs=100000,batch_size=3000, validation_split=0.1,verbose=2,
          callbacks=[es,mcp]
            )

#결과 예측
result=model.evaluate(x_test,y_test)
y_pred=model.predict([x_test])
print("loss:",result[0])
print("acc:",result[1])
# print("예측값:",y_pred)

'''
loss: 1.6934758424758911
acc: 0.5792520046234131
'''
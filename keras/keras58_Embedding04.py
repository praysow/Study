from keras.preprocessing.text import Tokenizer
import numpy as np
from keras.utils import pad_sequences
from keras.models import Sequential
from keras.layers import Dense,Conv1D,Flatten,LSTM,Embedding
from sklearn.model_selection import train_test_split
#1.데이터
docs = ['너무 재미있다','참 최고에요', '참 잘만든 영화에요','추천하고 싶은 영화입니다.','한 번 더 보고 싶어요.','글쎄',
        '별로에요', '생각보다 지루해요','연기가 어색해요','재미없어요','너무 재미없다.','참 재밋네요.','상헌이 바보', '반장 잘생겼다', '욱이 또 잔다']
labels = np.array([1,1,1,1,1,0,0,0,0,0,0,1,0,1,0])
token = Tokenizer()
token.fit_on_texts(docs)
# print(token.word_index)
# {'참': 1, '너무': 2, '재미있다': 3, '최고에요': 4, '잘만든': 5, '영화에요': 6, '추천하고': 7, '싶은': 8, '
# 영화입니다': 9, '한': 10, '번': 11, '더': 12, '보고': 13, '싶어요': 14, '글쎄': 15, '별로에요': 16, '생각보다': 17, '지루해요': 18, '연기가': 19, '어색해요': 20,
# '재미없어요': 21, '재미없다': 22, '재밋네요': 23, '상헌이': 24, '바보': 25, '반장': 26, '잘생겼다': 27, '욱이': 28, '또': 29, '잔다': 30} 단어 사전의 갯수는 30개
x = token.texts_to_sequences(docs)
y = labels
# print(y.shape)(15,)
# print(y)

# print(x)    [[2, 3], [1, 4], [1, 5, 6], [7, 8, 9], [10, 11, 12, 13, 14], [15], [16], [17, 18], [19, 20], [21], [2, 22], [1, 23], [24, 25], [26, 27], [28, 29, 30]]
# from keras.preprocessing.sequence import pad_sequences 안됨
# from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.utils import pad_sequences
# 시퀀스에 패딩을 적용하여 길이를 동일하게 만듭니다.
x = pad_sequences(x, padding='pre',
                  maxlen=5,#전체 길이를 조절
                  truncating='post' #앞이나 뒤를 잘라라(post 앞를 자름)
                  )

# x=np.array(x)
# # print(x)
# # print(x.shape)#(15, 5)
# #카테고리
# from keras.utils import to_categorical
# x1=to_categorical(x)
# # print(x1)
# # print(x1.shape)(15, 5, 31)
# #사이킷런
from sklearn.preprocessing import OneHotEncoder
x2=x.reshape(-1,1)
ohe = OneHotEncoder(sparse=False)
x2 = ohe.fit_transform(np.array(x).reshape(-1, 1)).reshape(15, 5, -1)
# # print(x2)
# # print(x2.shape) (15, 5, 31)
# #판다스
# import pandas as pd
# import numpy as np
# x3 = pd.get_dummies(x.reshape(-1)).to_numpy().reshape(15, 5, -1)
# print(x3)
# print(x3.shape)
#2.모델구성
x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.8,random_state=10)
# print(x_train.shape,y_train.shape)#(12, 5) (12,)
# print(x_test.shape,y_test.shape)#(3, 5) (3,)
x_train= x_train.reshape(12,5,1)
x_test= x_test.reshape(3,5,1)
#2.모델구성
model= Sequential()
model.add(Embedding(input_dim=30,output_dim=10
                    #,input_length=5
                    ))#input_dim=단어 사전의 갯수 output_dim=훈련의 갯수(노드수),input_length=shape의 갯수
model.add(LSTM(10))                          #임베딩계산식 = input_dim*output_dim
model.add(Dense(1,activation='sigmoid'))                       #엠베딩의 인풋의 shape " 2차원,임베딩 아웃풋의 shape:3차원"
model.summary()
########################################임베딩 1
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #
# =================================================================
#  embedding (Embedding)       (None, 5, 10)             310

#  lstm (LSTM)                 (None, 10)                840

#  dense (Dense)               (None, 1)                 11

# =================================================================
# Total params: 1,161
# Trainable params: 1,161
# Non-trainable params: 0
# _________________________________________________________________
#임베딩 2
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #
# =================================================================
#  embedding (Embedding)       (None, None, 10)          310

#  lstm (LSTM)                 (None, 10)                840

#  dense (Dense)               (None, 1)                 11

# =================================================================
# Total params: 1,161
# Trainable params: 1,161
# Non-trainable params: 0
# _________________________________________________________________
#임베딩 3
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #
# =================================================================
#  embedding (Embedding)       (None, None, 10)          300

#  lstm (LSTM)                 (None, 10)                840

#  dense (Dense)               (None, 1)                 11

# =================================================================
# Total params: 1,151
# Trainable params: 1,151
# Non-trainable params: 0 
# _________________________________________________________________

#3.컴파일 훈련
# model.compile(loss='binary_crossentropy',optimizer='adam',metrics='accuracy')
# model.fit(x_train,y_train,epochs=1000,batch_size=5,)
# #4.결과 예측
# result=model.evaluate(x_test,y_test)
# y_pred=model.predict([x])
# print("loss:",result[0])
# print("acc:",result[1])
# print("예측값:",y_pred)
#텐서플로우 서티피케이트의 4번 문제
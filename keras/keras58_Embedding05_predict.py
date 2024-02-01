from keras.preprocessing.text import Tokenizer
import numpy as np
from keras.utils import pad_sequences
from keras.models import Sequential
from keras.layers import Dense,Conv1D,Flatten,LSTM,Embedding
from sklearn.model_selection import train_test_split
#1.데이터
docs = ['너무 재미있다','참 최고에요', '참 잘만든 영화에요','추천하고 싶은 영화입니다.','한 번 더 보고 싶어요.','글쎄','나는 정룡이가 정말 싫다.',' 재미없다 너무 정말',
        '별로에요', '생각보다 지루해요','연기가 어색해요','재미없어요','너무 재미없다.','참 재밋네요.','상헌이 바보', '반장 잘생겼다', '욱이 또 잔다']
labels = np.array([1,1,1,1,1,0,0,0,0,0,0,0,0,1,0,1,0])
x_predict = '나는 정룡이가 정말 싫다. 재미없다 너무 정말'
token = Tokenizer()
token.fit_on_texts(docs)
token.fit_on_texts(x_predict)
x = token.texts_to_sequences(docs)
y = labels
test = token.texts_to_sequences(x_predict)

word_size = len(token.word_index)+1
print(word_size)

from keras.utils import pad_sequences
x = pad_sequences(x, padding='pre',
                  maxlen=5,#전체 길이를 조절
                  truncating='post' #앞이나 뒤를 잘라라(post 앞를 자름)
                  )
test = pad_sequences(test, padding='pre',
                  maxlen=5,#전체 길이를 조절
                  truncating='post' #앞이나 뒤를 잘라라(post 앞를 자름)
                  )
print(test.shape)
x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.8,random_state=10)
# print(x_train.shape,x_test.shape)

#2.모델구성
# x_train= x_train.reshape(13,5,1)
# x_test= x_test.reshape(4,5,1)
# test= test.reshape(25,5,1)
#2.모델구성
model= Sequential()
model.add(Embedding(input_dim=49,output_dim=1
                    #,input_length=5
                    ))#input_dim=단어 사전의 갯수 output_dim=훈련의 갯수(노드수),input_length=shape의 갯수
model.add(LSTM(10))
model.add(Dense(1))
model.add(Dense(1))
model.add(Dense(1,activation='sigmoid'))

#3.컴파일 훈련
model.compile(loss='binary_crossentropy',optimizer='adam',metrics='accuracy')
model.fit(x_train,y_train,epochs=1000,batch_size=5,)
#4.결과 예측
result=model.evaluate(x_test,y_test)
y_pred=model.predict([test])
print("loss:",result[0])
print("acc:",result[1])
print("예측값:",y_pred)

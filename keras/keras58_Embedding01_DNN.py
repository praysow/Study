from keras.preprocessing.text import Tokenizer
import numpy as np
from keras.utils import pad_sequences
from keras.models import Sequential
from keras.layers import Dense,Conv1D,Flatten
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
word_size = len(token.word_index)+1
print(word_size)
# print(x)    [[2, 3], [1, 4], [1, 5, 6], [7, 8, 9], [10, 11, 12, 13, 14], [15], [16], [17, 18], [19, 20], [21], [2, 22], [1, 23], [24, 25], [26, 27], [28, 29, 30]]
# from keras.preprocessing.sequence import pad_sequences 안됨
# # from tensorflow.keras.preprocessing.sequence import pad_sequences
# from keras.utils import pad_sequences
# # 시퀀스에 패딩을 적용하여 길이를 동일하게 만듭니다.
# x = pad_sequences(x, padding='pre',
#                   maxlen=5,#전체 길이를 조절
#                   truncating='post' #앞이나 뒤를 잘라라(post 앞를 자름)
#                   )
# # x=np.array(x)
# # # print(x)
# # print(x.shape)#(15, 5)
# # print(y.shape)

# #2.모델구성
# x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.8,random_state=10)

# print(x_train.shape,x_test.shape)
# print(y_train.shape,y_test.shape)

# model= Sequential()
# model.add(Dense(1,input_shape=(5,)))
# model.add(Dense(30))
# model.add(Dense(30))
# model.add(Dense(30))
# model.add(Dense(1,activation='sigmoid'))
# #3.컴파일 훈련
# model.compile(loss='binary_crossentropy',optimizer='adam',metrics='accuracy')
# model.fit(x_train,y_train,epochs=1000,batch_size=5,validation_split=0.2)
# #4.결과 예측
# result=model.evaluate(x_test,y_test)
# y_pred=model.predict([x])
# print("loss:",result)
# print("예측값:",y_pred)
# '''
# loss: [0.2731143534183502, 1.0]

# '''
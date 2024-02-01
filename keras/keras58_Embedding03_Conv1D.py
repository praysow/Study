from keras.preprocessing.text import Tokenizer
import numpy as np
from keras.utils import pad_sequences
from keras.models import Sequential
from keras.layers import Dense,Conv1D,Flatten,Conv2D
from sklearn.model_selection import train_test_split
#1.데이터
docs = ['너무 재미있다','참 최고에요', '참 잘만든 영화에요','추천하고 싶은 영화입니다.','한 번 더 보고 싶어요.','글쎄',
        '별로에요', '생각보다 지루해요','연기가 어색해요','재미없어요','너무 재미없다.','참 재밋네요.','상헌이 바보', '반장 잘생겼다', '욱이 또 잔다']
labels = np.array([1,1,1,1,1,0,0,0,0,0,0,1,0,1,0])
token = Tokenizer()
token.fit_on_texts(docs)

x = token.texts_to_sequences(docs)
y = labels

from keras.utils import pad_sequences
x = pad_sequences(x, padding='pre',
                  maxlen=5,#전체 길이를 조절
                  truncating='post' #앞이나 뒤를 잘라라(post 앞를 자름)
                  )


# #사이킷런
# from sklearn.preprocessing import OneHotEncoder
# x2=x.reshape(-1,1)
# ohe = OneHotEncoder(sparse=False)
# x2 = ohe.fit_transform(np.array(x).reshape(-1, 1)).reshape(15, 5, -1)

#2.모델구성
x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.8,random_state=10)

print(x_train.shape,y_train.shape)#(12, 5) (12,)
print(x_test.shape,y_test.shape)#(3, 5) (3,)
x_train= x_train.reshape(12,5,1)
x_test= x_test.reshape(3,5,1)


model= Sequential()
model.add(Conv1D(filters=10,kernel_size=2,input_shape=(5,1)))
# model.add(Conv2D(10,(2,1),input_shape=(5,1,1)))
model.add(Flatten())
model.add(Dense(1,activation='sigmoid'))
#3.컴파일 훈련
model.compile(loss='binary_crossentropy',optimizer='adam',metrics='accuracy')
model.fit(x_train,y_train,epochs=1000,batch_size=5,validation_split=0.2)
#4.결과 예측
result=model.evaluate(x_test,y_test)
y_pred=model.predict([x])
print("loss:",result[0])
print("acc:",result[1])
print("예측값:",y_pred)
'''
loss: 0.20409266650676727
acc: 1.0
'''
from keras.preprocessing.text import Tokenizer
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
test = '나는 진짜 진짜 매우 매우 맛있는 밥을 엄청 마구 마구 마구 먹었다'
test2 = '상헌이가 선생을 괴롭힌다. 상헌이는 못생겼다. 상헌이는 마구 마구 못생겼다.'

token = Tokenizer()
token.fit_on_texts([test])
token.fit_on_texts([test2])

# print(token.word_index)
# print(token.word_counts)

x=token.texts_to_sequences([test])
x_2=token.texts_to_sequences([test2])

# print(x)
from keras.utils import to_categorical
x1=to_categorical(x)
x1_2=to_categorical(x_2)
# #1. to_categorical 에서 첫번째 0빼
x1 = x1[:,:,1:]
x1_2 = x1_2[:,:,1:] #(1, 9, 13)

# print(x1)
# print(x1.shape)  #(1, 12, 8)

#2. 사이킷런 원핫인코더
import numpy as np
# ohe = OneHotEncoder(sparse=False)
# x2 = ohe.fit_transform(np.array(x).reshape(-1, 1)).reshape(1, 12, -1)
# x2_2 = ohe.fit_transform(np.array(x_2).reshape(-1, 1)).reshape(1, 9, -1)
# #3.판다스 겟더미
# x3 = pd.get_dummies(x.reshape(-1)).to_numpy().reshape(1, 12, -1)
x3 = pd.get_dummies(np.array(x).reshape(-1)).to_numpy().reshape(1, 12, -1)
x3_2 = pd.get_dummies(np.array(x_2).reshape(-1)).to_numpy().reshape(1, 9, -1)

print(x3_2)
print(x3_2.shape)

from keras.preprocessing.text import Tokenizer
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
test = '나는 진짜 진짜 매우 매우 맛있는 밥을 엄청 마구 마구 마구 먹었다'

token = Tokenizer()
token.fit_on_texts([test])

# print(token.word_index)
# print(token.word_counts)

x=token.texts_to_sequences([test])
# print(x)

from keras.utils import to_categorical
x1=to_categorical(x)

# print(x)
# print(x.shape)  #(1, 12, 9)

#1. to_categorical 에서 첫번째 0빼
# x1 = x1[:,:,1:]
# print(x1)
# print(x1.shape)  #(1, 12, 8)

#2. 사이킷런 원핫인코더
import numpy as np
# x = [item for sublist in x for item in sublist]
# x = np.array(x).reshape(-1, 1)
# ohe = OneHotEncoder(sparse=False)
# x_ohe = ohe.fit_transform(x)
# x = pd.DataFrame(x_ohe)
# x = x.values.reshape(1, 12, 8)


# #3.판다스 겟더미
x = [item for sublist in x for item in sublist]     #list화 시키기
x2 = pd.get_dummies(x)
x2 = pd.DataFrame(x2)
x2 = x2.values.reshape(1, 12, 8)
print(x2)
print(x2.shape)

# Flatten 대신 GlobalAveragePooling 을 사용할 수 있다.
# Flatten 의 문제점이 있다 (너무 크다 = 연산량이 많다)
import numpy as np
from keras.datasets import mnist
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense , Conv2D , Flatten  , MaxPooling2D , GlobalAveragePooling2D      # Flatten : 평평한

#1 데이터
(x_train , y_train), (x_test, y_test)  =  mnist.load_data()
print(x_train.shape , y_train.shape)    # (60000, 28, 28) (60000,)
print(x_test.shape , y_test.shape)      # (10000, 28, 28) (10000,)

print(x_train)
print(x_train[0])
print(np.unique(y_train, return_counts=True))
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949],dtype=int64))
print(pd.value_counts(y_test))

x_train = x_train.reshape(60000,28,28,1)
x_test = x_test.reshape(10000,28,28,1)
# print(x_train.shape[0])         # 60000

# x_test = x_test.reshape(x_test[0],x_test[1],x_test[2],1)        # 위에 10000,28,28,1 이랑 같다. 이렇게 쓰는게 나중에 전처리하고 test가 달라졌을 때 좋을수도 있다.

print(x_train.shape , x_test.shape)     # (60000, 28, 28, 1) (10000, 28, 28, 1)



#2 모델구성
model = Sequential()
model.add(Conv2D(100,(2,2),input_shape = (10,10,1), padding='same' , strides= 1 ))        # 10 = 필터 / (2,2) = 커널 사이즈  // strides = 보폭의 크기 // padding = 'valid' 디폴트
#                              shape = (batch_size(model.fit에 들어가는 batch_size // 행이랑 똑같다),rows,columns,channels)
#                              shape = (batch_size,heights,widths,channels)
model.add(MaxPooling2D())       # (None,5,5,100)
model.add(Conv2D(filters = 100,kernel_size = (2,2)))    # (None,4,4,100)
model.add(Conv2D(100,(2,2)))    # (None,3,3,100)
model.add(Flatten())
# model.add(GlobalAveragePooling2D())     # (None,100)
model.add(Dense(units=50))              # (None,50) 5050
model.add(Dense(10,activation= 'softmax'))  #(None,10) 510

model.summary()

# Globalaveragepooling
# Total params: 86,260
# Trainable params: 86,260
# Non-trainable params: 0

# Flatten
# Total params: 126,260
# Trainable params: 126,260
# Non-trainable params: 0


""" #3 컴파일, 훈련
model.compile(loss = 'categorical_crossentropy' , optimizer = 'adam' , metrics = ['acc'] )
model.fit(x_train,y_train , epochs = 100 , batch_size=32 , verbose=1 , validation_split= 0.2 )


#4 평가, 예측
result = model.evaluate(x_test, y_test)
print('loss = ',result[0])
print('acc = ',result[1])
 """
# 오류가 나는 이유 // Shapes (32,) and (32, 27, 27, 10) are incompatible = 호환되지 않는다. 32, 와 32, 27, 27, 10 가 호환 X
# (32,) = 1차원 (32,27,27,10) = 4차원 이라서 오류가 발생한다.
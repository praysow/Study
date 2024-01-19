'''
import numpy as np
import pandas as pd
from keras.datasets import  mnist
from keras.layers import Dense,Conv2D
from keras.models import Sequential
(x_train, y_train),(x_test,y_test ) =mnist.load_data()
# print(x_train[28])
# print(np.unique(x_train,return_counts=True))
# print(pd.value_counts(y_train))
# print(x_train.shape, y_train.shape) (60000, 28, 28) (60000,)
# print(x_test.shape, y_test.shape)   (10000, 28, 28) (10000,)
#다음은 리쉐이프 해주기!!

x_train = x_train.reshape(60000,28,28,1)        #리세이프는 데이터의 내용과 순서가 바뀌지 않으면 가능하다
#x_test = x_test.reshape(10000,28,28,1)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],x_test.shape[2],1)

# print(x_train.shape,x_test.shape)       (60000, 28, 28, 1) (10000, 28, 28, 1)

#2.모델구성
model =Sequential()
model.add(Conv2D(4, (2,2),input_shape=(28,28,1)))                  #배치크기만 지정하고 이미지의 차원은 지정을 안함!!!
model.add(Dense(1))                                     #Dence는 1D밖에 못받는다 그러므로 Conv2D로 된 이미지들을 Flatten()을 사용하여 다차원을 1D로 바꿔줘야한다
model.add(Dense(1))
model.add(Dense(1))
model.add(Dense(1))
model.add(Dense(10, activation='softmax'))

#3.컴파일 훈련
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(x_train,y_train,epochs=100,batch_size=32, verbose=1,validation_split=0.2)

#4. 평가예측
result=model.evaluate(x_test,y_test)
print("loss",result[0])
print("acc",result[0])

'''

import numpy as np
import pandas as pd
from keras.datasets import mnist
from keras.layers import Dense, Conv2D, Flatten     #Flatten: 평탄화시키다
from keras.models import Sequential
from keras.utils import to_categorical
import time
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

model = Sequential()
model.add(Conv2D(1, (2, 2), input_shape=(28, 28, 1)))      #(2,2)를 사용했더니 (28,28,1)이 input의 4를 더해서 (n,27,27,4)가 된다
model.add(Conv2D(2, (3, 3)))
model.add(Conv2D(3, (4, 4)))
# model.add(Flatten())
# model.add(Dense(20))
# model.add(Dense(20))
# model.add(Dense(10))
# model.add(Dense(10))
# model.add(Dense(10, activation='softmax'))

model.summary()


# 모델의 입력 형태를 (28, 28, 1)로 설정했지만, model.fit에서는 4D 텐서를 전달하는데, 이로 인해 모델과 데이터의 차원이 맞지 않아서 발생한 오류입니다.
# 모델의 입력 형태를 맞추기 위해 Flatten() 레이어를 추가하여 4D 텐서를 1D로 펼치고, 레이블을 원-핫 인코딩으로 변환하여 해결하였습니다.
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# start_time = time.time()
# model.fit(x_train, y_train, epochs=100, batch_size=32, verbose=1, validation_split=0.2)
# end_time = time.time()

# result = model.evaluate(x_test, y_test)
# print("loss", result[0])
# print("acc", result[1])
# print("걸린시간 :",round(end_time - start_time))


'''
loss 0.32011738419532776
acc 0.916100025177002
걸린시간 : 252
'''
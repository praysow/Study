import numpy as np
import pandas as pd
from keras.datasets import mnist
from keras.layers import *
from keras.models import Sequential
from keras.utils import to_categorical
import time
from keras.optimizers import Adam
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape    (60000, 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

model = Sequential()
model.add(Conv2D(10, (2, 2), input_shape=(28, 28, 1)))      #(2,2)를 사용했더니 (28,28,1)이 input의 4를 더해서 (n,27,27,4)가 된다
model.add(Conv2D(20, (3, 3),activation='relu'))
model.add(Conv2D(30, (4, 4),activation='relu'))
model.add(Conv2D(30, (4, 4),activation='relu'))
model.add(Conv2D(30, (4, 4),activation='relu'))
# model.add(Flatten())
model.add(GlobalAveragePooling2D())
model.add(Dense(40,activation='relu'))
model.add(Dense(30,activation='relu'))
model.add(Dense(20,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(10, activation='softmax'))




# 모델의 입력 형태를 (28, 28, 1)로 설정했지만, model.fit에서는 4D 텐서를 전달하는데, 이로 인해 모델과 데이터의 차원이 맞지 않아서 발생한 오류입니다.
# 모델의 입력 형태를 맞추기 위해 Flatten() 레이어를 추가하여 4D 텐서를 1D로 펼치고, 레이블을 원-핫 인코딩으로 변환하여 해결하였습니다.
from keras.callbacks import EarlyStopping,ModelCheckpoint,ReduceLROnPlateau
es= EarlyStopping(monitor='val_loss',mode='auto',patience=50,verbose=1,restore_best_weights=True)
mcp = ModelCheckpoint(monitor='val_loss',mode='auto',verbose=1,save_best_only=True,
                      filepath='../_data/_save/MCP/keras31-6.hdf5')
rlr = ReduceLROnPlateau(monitor='val_loss',patience=10,mode='auto',verbose=1,factor=0.5)
lr = 0.0001
model.compile(loss='categorical_crossentropy',optimizer=Adam(learning_rate=lr),metrics=['accuracy'])
start_t = time.time()
model.fit(x_train,y_train,epochs=20, batch_size=10, verbose=2, validation_split=0.1,callbacks=[es,mcp])
end_t= time.time()

result = model.evaluate(x_test, y_test)
print("loss", result[0])
print("acc", result[1])
print("걸린시간 :",round(end_t - start_t))

'''
loss 0.9844655990600586
acc 0.661899983882904
걸린시간 : 240

loss 0.03718910366296768
acc 0.9884999990463257
걸린시간 : 301
'''
import time
import numpy as np
import pandas as pd
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Dense, Conv2D, AveragePooling2D,BatchNormalization,Dropout,Flatten,MaxPooling2D
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

(x_train,y_train),(x_test,y_test) = fashion_mnist.load_data()

# print(x_train.shape,y_train.shape)  #(60000, 28, 28) (60000,)
# print(x_test.shape,y_test.shape)    #(10000, 28, 28) (10000,)
# print(pd.value_counts(y_train))
# print(np.unique(x_train,return_counts=True))

x_train,x_test,y_train,y_test=train_test_split(x_train,y_train,train_size=0.9,random_state=3)

#1. 데이터
# x_train= x_train/ 255.0
# y_train= y_train/ 255.0
x_train= x_train.reshape(x_train.shape[0],x_train.shape[1],x_train.shape[2],1)
x_test= x_test.reshape(x_test.shape[0], x_test.shape[1],x_test.shape[2],1)

y_train =to_categorical(y_train,10)
y_test = to_categorical(y_test, 10)

from keras import regularizers


#2.모델생성
model = Sequential()
model.add(Conv2D(32,(3,3), input_shape=(28,28,1),strides=1,padding='same'))
model.add(MaxPooling2D())
model.add(Dropout(0.5))
model.add(Conv2D(64,(3,3), activation='relu'))
model.add(AveragePooling2D())
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dense(256,activation='relu'))
model.add(Dense(512,activation='relu'))
model.add(BatchNormalization())
model.add(Dense(256,activation='relu',kernel_regularizer=regularizers.l2(0.01)))
model.add(Dense(10,activation='softmax'))

from keras.callbacks import EarlyStopping,ModelCheckpoint
es= EarlyStopping(monitor='val_loss',mode='auto',patience=100,verbose=1,restore_best_weights=True)
mcp = ModelCheckpoint(monitor='val_loss',mode='auto',verbose=1,save_best_only=True,
                      filepath='../_data/_save/MCP/keras31-7.hdf5')
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
start_t = time.time()
model.fit(x_train,y_train,epochs=5, batch_size=32, verbose=2, validation_split=0.1,callbacks=[es,mcp])
end_t= time.time()

# 모델 평가
result = model.evaluate(x_test, y_test)
y_submit= model.predict(x_test)
y_test_indices = np.argmax(y_test, axis=1)
y_submit_indices = np.argmax(y_submit, axis=1)

print("Loss:", result[0])
print("Accuracy:", result[1])
print("걸린 시간:", round(end_t - start_t))


'''
Loss: 0.40834110975265503
Accuracy: 0.8243333101272583
걸린 시간: 20

'''
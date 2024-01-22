
import time
import numpy as np
from keras.datasets import cifar100
from keras.models import Sequential
from keras.layers import Dense,Conv3D,Conv2D,BatchNormalization,AveragePooling3D, MaxPooling2D,Dropout,Flatten,MaxPooling3D
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import pandas as pd
(x_train, y_train), (x_test,y_test) = cifar100.load_data()


# print(x_train.shape,y_train.shape)  #(50000, 32, 32, 3) (50000, 1)
# print(x_test.shape,y_test.shape)   # (10000, 32, 32, 3) (10000, 1)
# print(pd.value_counts(y_train))
# print(np.unique(y_train,return_counts=True))


x_train,x_test,y_train,y_test = train_test_split(x_train,y_train,train_size=0.9,random_state=3)

#1.데이터 전처리

x_train = x_train.astype('float32') /255.0
x_test = x_test.astype('float32') /255.0
# x_train = x_train.reshape(50000,32,32,3)
# x_test = x_test.reshape(10000,32,32,3)
x_train =x_train.reshape(x_train.shape[0],x_train.shape[1],x_train.shape[2],x_train.shape[3])
x_test =x_test.reshape(x_test.shape[0], x_test.shape[1],x_test.shape[2],x_test.shape[3])
x_train = to_categorical(x_train,100)
y_test = to_categorical(x_test,100)






#2. 모델생성 
model = Sequential()
model.add(Conv3D(32, (3,3,3),input_shape=(32,32,3,2),activation='relu'))
model.add(AveragePooling3D(pool_size=(1,1,1)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(30,activation='relu'))
model.add(Dense(100,activation='softmax'))

from keras.callbacks import EarlyStopping,ModelCheckpoint
es= EarlyStopping(monitor='val_loss',mode='auto',patience=100,verbose=1,restore_best_weights=True)
mcp = ModelCheckpoint(monitor='val_loss',mode='auto',verbose=1,save_best_only=True,
                      filepath='../_data/_save/MCP/keras31-1.hdf5')
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
start_t = time.time()
model.fit(x_train,y_train,epochs=10, batch_size=32, verbose=2, validation_split=0.1,callbacks=[es,mcp])
end_t= time.time()

# 모델 평가
result = model.evaluate(x_test, y_test)
y_submit= model.predict(x_test)
y_test_indices = np.argmax(y_test, axis=1)
y_submit_indices = np.argmax(y_submit, axis=1)

print("Loss:", result[0])
print("Accuracy:", result[1])
print("걸린 시간:", round(end_t - start_t))

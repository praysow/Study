import time
import numpy as np
from keras.datasets import cifar100
from keras.models import Sequential
from keras.layers import *
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import pandas as pd
from keras.optimizers import Adam
(x_train, y_train), (x_test,y_test) = cifar100.load_data()

x_train,x_test,y_train,y_test = train_test_split(x_train,y_train,train_size=0.9,random_state=3)

x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
y_train = to_categorical(y_train, 100)
y_test = to_categorical(y_test, 100)

#2. 모델생성 
model = Sequential()
model.add(Conv2D(32, (3,3),input_shape=(32,32,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.5))
model.add(Conv2D(64, (3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))
model.add(Conv2D(128, (3,3),activation='relu'))
model.add(AveragePooling2D(pool_size=(2,2)))
model.add(Dropout(0.3))
# model.add(Flatten())
model.add(GlobalAveragePooling2D())
model.add(Dense(512,activation='relu'))
model.add(Dense(256,activation='relu'))
model.add(BatchNormalization())
model.add(Dense(128,activation='relu'))
model.add(Dense(100,activation='softmax'))

#3.컴파일 훈련
from keras.callbacks import EarlyStopping,ModelCheckpoint,ReduceLROnPlateau
es= EarlyStopping(monitor='val_loss',mode='auto',patience=50,verbose=1,restore_best_weights=True)
mcp = ModelCheckpoint(monitor='val_loss',mode='auto',verbose=1,save_best_only=True,
                      filepath='../_data/_save/MCP/keras31-6.hdf5')
rlr = ReduceLROnPlateau(monitor='val_loss',patience=10,mode='auto',verbose=1,factor=0.5)
lr = 0.0001
model.compile(loss='categorical_crossentropy',optimizer=Adam(learning_rate=lr),metrics=['accuracy'])
start_t = time.time()
model.fit(x_train,y_train,epochs=100, batch_size=10, verbose=2, validation_split=0.1,callbacks=[es,mcp])
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
Loss: 2.4665298461914062
Accuracy: 0.3864000141620636
걸린 시간: 272

Loss: 2.477313280105591
Accuracy: 0.39980000257492065
걸린 시간: 1222

'''
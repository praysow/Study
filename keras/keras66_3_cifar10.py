import time
import numpy as np
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import *
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import pandas as pd
from keras.optimizers import Adam
(x_train, y_train), (x_test, y_test) = cifar10.load_data()


# print(pd.value_counts(y_train))
# print(np.unique(x_train,return_counts=True))

x_train,x_test,y_train,y_test=train_test_split(x_train,y_train,train_size=0.9,random_state=3)




# 데이터 전처리
x_train = x_train / 255.0
x_test = x_test / 255.0
x_train =x_train.reshape(x_train.shape[0],x_train.shape[1],x_train.shape[2],x_train.shape[3])
x_test =x_test.reshape(x_test.shape[0], x_test.shape[1],x_test.shape[2],x_test.shape[3])
y_train = to_categorical(y_train,10)
y_test = to_categorical(y_test,10)

# 모델 생성
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(32, 32, 3), activation='relu'))
model.add(AveragePooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(AveragePooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(AveragePooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
# model.add(Flatten())
model.add(GlobalAveragePooling2D())
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(256, activation='relu'))
model.add(Dense(10, activation='softmax'))

# 모델 컴파일 및 학습
from keras.callbacks import EarlyStopping,ModelCheckpoint,ReduceLROnPlateau
es= EarlyStopping(monitor='val_loss',mode='auto',patience=50,verbose=1,restore_best_weights=True)
mcp = ModelCheckpoint(monitor='val_loss',mode='auto',verbose=1,save_best_only=True,
                      filepath='../_data/_save/MCP/keras31-6.hdf5')
rlr = ReduceLROnPlateau(monitor='val_loss',patience=1,mode='auto',verbose=1,factor=0.5)
lr = 0.001
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

Loss: 0.7987858057022095
Accuracy: 0.7537999749183655
걸린 시간: 82

Loss: 0.8796550035476685
Accuracy: 0.7027999758720398
걸린 시간: 881
'''


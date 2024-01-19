import time
import numpy as np
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, BatchNormalization, MaxPooling2D,Dropout,AveragePooling2D
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train,x_test,y_train,y_test=train_test_split(x_train,y_train,train_size=0.9,random_state=3)

# 데이터 전처리
x_train = x_train / 255.0
x_test = x_test / 255.0
x_train =x_train.reshape(x_train.shape[0],x_test.shape[1],x_test.shape[2],x_test.shape[3])
x_test =x_test.reshape(x_test.shape[0], x_test.shape[1],x_test.shape[2],x_test.shape[3])
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

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
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(256, activation='relu'))
model.add(Dense(10, activation='softmax'))

# 모델 컴파일 및 학습
from keras.callbacks import EarlyStopping,ModelCheckpoint
es= EarlyStopping(monitor='val_loss',mode='auto',patience=100,verbose=1,restore_best_weights=True)
mcp = ModelCheckpoint(monitor='val_loss',mode='auto',verbose=1,save_best_only=True,
                      filepath='../_data/_save/MCP/keras31-1.hdf5')
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
start_t = time.time()
model.fit(x_train,y_train,epochs=1000, batch_size=500, verbose=2, validation_split=0.1,callbacks=[es,mcp])
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

Loss: 0.6911240816116333
Accuracy: 0.7680000066757202
걸린 시간: 120
'''


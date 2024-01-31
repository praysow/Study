import time
import numpy as np
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, LSTM,Conv1D,Flatten
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import pandas as pd
(x_train, y_train), (x_test, y_test) = cifar10.load_data()


# print(pd.value_counts(y_train))
# print(np.unique(x_train,return_counts=True))

x_train,x_test,y_train,y_test=train_test_split(x_train,y_train,train_size=0.9,random_state=3)




# 데이터 전처리
x_train = x_train / 255.0
x_test = x_test / 255.0
x_train =x_train.reshape(x_train.shape[0],x_train.shape[1],x_train.shape[2]*x_train.shape[3])
x_test =x_test.reshape(x_test.shape[0], x_test.shape[1],x_test.shape[2]*x_test.shape[3])
y_train = to_categorical(y_train,10)
y_test = to_categorical(y_test,10)

# 모델 생성
model = Sequential()
model.add(Conv1D(filters=32,kernel_size=2, input_shape=(32, 32*3), activation='relu'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(10, activation='softmax'))

# 모델 컴파일 및 학습
from keras.callbacks import EarlyStopping,ModelCheckpoint
es= EarlyStopping(monitor='val_loss',mode='auto',patience=100,verbose=1,restore_best_weights=True)
mcp = ModelCheckpoint(monitor='val_loss',mode='auto',verbose=1,save_best_only=True,
                      filepath='../_data/_save/MCP/keras31-1.hdf5')
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
start_t = time.time()
model.fit(x_train,y_train,epochs=1000, batch_size=32, verbose=2, validation_split=0.1,callbacks=[es,mcp])
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

Loss: 1.3617868423461914
Accuracy: 0.5216000080108643
걸린 시간: 4458

Loss: 1.344199538230896
Accuracy: 0.5351999998092651
걸린 시간: 203
'''


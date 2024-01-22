import time
import numpy as np
from keras.datasets import cifar100
from keras.models import Sequential
from keras.layers import Dense, Conv2D,BatchNormalization, AveragePooling2D,Dropout,Flatten, MaxPooling2D
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import pandas as pd
(x_train, y_train), (x_test,y_test) = cifar100.load_data()


# print(x_train.shape,y_train.shape)  #(50000, 32, 32, 3) (50000, 1)
# print(x_test.shape,y_test.shape)   # (10000, 32, 32, 3) (10000, 1)

x_train,x_test,y_train,y_test = train_test_split(x_train,y_train,train_size=0.9,random_state=3)

# print(np.unique(x_train,return_counts=True))

x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
x_train =x_train.reshape(x_train.shape[0],x_train.shape[1]*x_train.shape[2]*x_train.shape[3])
x_test =x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2]*x_test.shape[3])

y_train = to_categorical(y_train, 100)
y_test = to_categorical(y_test, 100)

#2. 모델생성 
model = Sequential()
model.add(Dense(256,input_shape=(3072,)))
model.add(Dense(512,activation='relu'))
model.add(Dense(256,activation='relu'))
model.add(BatchNormalization())
model.add(Dense(128,activation='relu'))
model.add(Dense(100,activation='softmax'))

#3.컴파일 훈련
from keras.callbacks import EarlyStopping,ModelCheckpoint
es= EarlyStopping(monitor='val_loss',mode='auto',patience=50,verbose=1,restore_best_weights=True)
mcp = ModelCheckpoint(monitor='val_loss',mode='auto',verbose=1,save_best_only=True,
                      filepath='../_data/_save/MCP/keras31-6.hdf5')
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
start_t = time.time()
model.fit(x_train,y_train,epochs=500, batch_size=10, verbose=2, validation_split=0.1,callbacks=[es,mcp])
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

Loss: 2.3619656562805176
Accuracy: 0.41200000047683716
걸린 시간: 1327

Loss: 3.305086374282837
Accuracy: 0.23919999599456787
걸린 시간: 634
'''
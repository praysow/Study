import time
import numpy as np
import pandas as pd
from keras.datasets import fashion_mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, AveragePooling2D,BatchNormalization,Dropout,Flatten,MaxPooling2D, Input
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

(x_train,y_train),(x_test,y_test) = fashion_mnist.load_data()

# ### 스케일링 1-1
# x_train = x_train /255           # minmaxscaler랑 비슷
# x_test = x_test /255

### 스게일링 1-2
# x_train = (x_train - 127.5)/127.5           #중위값을 맞춰주는것
# x_test = (x_test - 127.5)/127.5             #standrad랑 비슷

x_train = x_train.reshape (60000, 28*28)
x_test = x_test.reshape(10000,28*28)
### 스케일링 2-1
# scaler = MinMaxScaler()
# x_train = scaler.fit_transform(x_train)
# x_test= scaler.transform(x_test)
# ### 스케일링 2-2
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test= scaler.transform(x_test)
# print(x_train.shape,y_train.shape)  #(60000, 28, 28) (60000,)
# print(x_test.shape,y_test.shape)    #(10000, 28, 28) (10000,)
# print(pd.value_counts(y_train))
# print(np.unique(x_train,return_counts=True))

x_train,x_test,y_train,y_test=train_test_split(x_train,y_train,train_size=0.9,random_state=3)

#1. 데이터
# x_train= x_train/ 255.0
# y_train= y_train/ 255.0
x_train= x_train.reshape(x_train.shape[0],28,28,1)
x_test= x_test.reshape(x_test.shape[0],28,28,1)

y_train =to_categorical(y_train,10)
y_test = to_categorical(y_test, 10)

from keras import regularizers


#2.모델생성

model = Sequential()
input1= Input(shape=(28,28,1))
conv1 = Conv2D(32,(3,3),activation='relu')(input1)
max1 = MaxPooling2D()(conv1)
drop1 = Dropout(0.5)(max1)
conv2 = Conv2D(64,(3,3),activation='relu')(conv1)
max2 = AveragePooling2D()(conv2)
drop2 = Dropout(0.5)(max2)
flat1=Flatten()(drop2)
dense2= Dense(128,activation='relu')(flat1)
dense3= Dense(256,activation='relu')(dense2)
dense4= Dense(512,activation='relu')(dense3)
dense5= Dense(256,activation='relu')(dense4)
bat1=BatchNormalization()(dense5)
output= Dense(10,activation='softmax')(bat1)
model = Model(inputs=input1,outputs=output)








from keras.callbacks import EarlyStopping,ModelCheckpoint
es= EarlyStopping(monitor='val_loss',mode='auto',patience=100,verbose=1,restore_best_weights=True)
mcp = ModelCheckpoint(monitor='val_loss',mode='auto',verbose=1,save_best_only=True,
                      filepath='../_data/_save/MCP/keras31-7.hdf5')
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
start_t = time.time()
model.fit(x_train,y_train,epochs=100, batch_size=32, verbose=2, validation_split=0.1,callbacks=[es,mcp])
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
Loss: 0.3514007031917572
Accuracy: 0.9381666779518127
걸린 시간: 335

Loss: 0.3217534124851227
Accuracy: 0.9365000128746033
걸린 시간: 337

Loss: 0.32816407084465027
Accuracy: 0.9359999895095825
걸린 시간: 337


Loss: 0.32696691155433655
Accuracy: 0.9359999895095825
걸린 시간: 339
'''
import time
import numpy as np
from keras.datasets import cifar100
from keras.models import Sequential,Model
from keras.layers import Dense, Conv2D,BatchNormalization, AveragePooling2D,Dropout,Flatten, MaxPooling2D,Input
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

(x_train, y_train), (x_test,y_test) = cifar100.load_data()

# ### 스케일링 1-1
# x_train = x_train /255            minmaxscaler랑 비슷
# x_test = x_test /255

### 스게일링 1-2
# x_train = (x_train - 127.5)/127.5           #중위값을 맞춰주는것
# x_test = (x_test - 127.5)/127.5             #standrad랑 비슷

x_train = x_train.reshape(x_train.shape[0],32*32*3)
x_test = x_test.reshape(x_test.shape[0],32*32*3)
### 스케일링 2-1
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test= scaler.transform(x_test)
# ### 스케일링 2-2
# scaler = StandardScaler()
# x_train = scaler.fit_transform(x_train)
# x_test= scaler.transform(x_test)
# print(x_train.shape,y_train.shape)  #(50000, 32, 32, 3) (50000, 1)
# print(x_test.shape,y_test.shape)   # (10000, 32, 32, 3) (10000, 1)

x_train,x_test,y_train,y_test = train_test_split(x_train,y_train,train_size=0.9,random_state=3)

# print(np.unique(x_train,return_counts=True))

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train =x_train.reshape(x_train.shape[0],32,32,3)
x_test =x_test.reshape(x_test.shape[0],32,32,3)

y_train = to_categorical(y_train, 100)
y_test = to_categorical(y_test, 100)

#2. 모델생성 
model = Sequential()
input1= Input(shape=(32,32,3))
conv1 = Conv2D(32,(3,3),activation='relu')(input1)
max1 = MaxPooling2D()(conv1)
drop1 = Dropout(0.5)(max1)
conv2 = Conv2D(64,(3,3),activation='relu')(conv1)
max2 = MaxPooling2D()(conv2)
drop2 = Dropout(0.5)(max2)
conv3 = Conv2D(128,(4,4),activation='relu')(drop2)
max3 = MaxPooling2D()(conv3)
drop3 = Dropout(0.5)(max3)
flat1=Flatten()(drop3)
dense2= Dense(512,activation='relu')(flat1)
dense3= Dense(256,activation='relu')(dense2)
output= Dense(100,activation='softmax')(dense3)
model = Model(inputs=input1,outputs=output)

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
Loss: 2.9752931594848633
Accuracy: 0.28600001335144043
걸린 시간: 900

Loss: 2.7811219692230225
Accuracy: 0.34060001373291016
걸린 시간: 911

Loss: 3.104231119155884
Accuracy: 0.26260000467300415
걸린 시간: 478

Loss: 2.58732271194458
Accuracy: 0.38040000200271606
걸린 시간: 1948
'''
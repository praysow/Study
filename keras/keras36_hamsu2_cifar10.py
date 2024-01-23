import time
import numpy as np
from keras.datasets import cifar10
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Flatten, BatchNormalization, MaxPooling2D,Dropout,AveragePooling2D, Input
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# ### 스케일링 1-1
# x_train = x_train /255            minmaxscaler랑 비슷
# x_test = x_test /255

### 스게일링 1-2
# x_train = (x_train - 127.5)/127.5           #중위값을 맞춰주는것
# x_test = (x_test - 127.5)/127.5             #standrad랑 비슷

x_train = x_train.reshape    (x_train.shape[0],32*32*3)
x_test = x_test.reshape(x_test.shape[0],32*32*3)
### 스케일링 2-1
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test= scaler.transform(x_test)
# ### 스케일링 2-2
# scaler = StandardScaler()
# x_train = scaler.fit_transform(x_train)
# x_test= scaler.transform(x_test)

# print(pd.value_counts(y_train))
# print(np.unique(x_train,return_counts=True))

x_train,x_test,y_train,y_test=train_test_split(x_train,y_train,train_size=0.9,random_state=3)




# 데이터 전처리
x_train =x_train.reshape(x_train.shape[0],32,32,3)
x_test =x_test.reshape(x_test.shape[0],32,32,3)
y_train = to_categorical(y_train,10)
y_test = to_categorical(y_test,10)

# 모델 생성
odel = Sequential()
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
output= Dense(10,activation='softmax')(dense3)
model = Model(inputs=input1,outputs=output)





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
Loss: 0.7342924475669861
Accuracy: 0.7559999823570251
걸린 시간: 496

Loss: 0.7005064487457275
Accuracy: 0.7721999883651733
걸린 시간: 362

Loss: 0.7207982540130615
Accuracy: 0.7498000264167786
걸린 시간: 332

Loss: 0.7223435044288635
Accuracy: 0.7680000066757202
걸린 시간: 538
'''


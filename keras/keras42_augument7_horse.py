from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense,Conv2D,MaxPooling2D,Dropout,Flatten,BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
from sklearn.model_selection import train_test_split
train_datagen=ImageDataGenerator(
    #  rescale=1./255,                      #앞에 1.에서 .은 부동소수점 계산하겠다는 것이다
     horizontal_flip=True,           #수평 뒤집기
     vertical_flip=True,             #수직 뒤집기
     width_shift_range=0.1,           #평행이동 0.1=10%이동
     height_shift_range=0.1,          #높이방향으로 이동
     rotation_range=5,               #정해진 각도 만큼 이미지를 회전 (5도 회전)
     zoom_range=1.2,                 #1.2배 확대혹은 축소
     shear_range=0.9,                 #좌표 하나를 고정시키고 다른 몇개의 좌표를 이동시키는 변환(마름모형태로 만듬)
     fill_mode='nearest'
     )

test_datagen = ImageDataGenerator(rescale=1./255)
path_train= "c:/_data/image/horse_human/train/"
# print(xy_train)
test=test_datagen.flow_from_directory(path_train,target_size=(300,300),batch_size=100,class_mode='binary')            #원본데이터는 최대한 건드리지 말자,원본데이터는 각각다르니 target_size를 통해서 사이즈를 동일화시킨다
# print(np.unique(xy_train,return_counts=True))

np_path='c:/_data/_save_npy/'
x= np.load(np_path + 'keras39_7_x_train.npy')
y= np.load(np_path + 'keras39_7_y_train.npy')

x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.8,random_state=1)

augumet_size = 10000

randidx= np.random.randint(x_train.shape[0],size=augumet_size)  #np.random.randint(60000,40000) 6만개의 데이터중 4만개의 숫자를 뽑아내라

x_augumented = x_train[randidx].copy()      #그냥 집어넣게 되면은 x_train이 수정되는 경우가 있어서 안전하게 copy로 x_augumented를 정의 한다
y_augumented = y_train[randidx].copy()
x_augumented= x_augumented.reshape(-1,300,300,1)


x_augumented = train_datagen.flow(
    x_augumented,
    y_augumented,
    batch_size=augumet_size,
    shuffle=False,
    ).next()[0]


x_train = x_train.reshape(-1,300,300,1)
x_test = x_test.reshape(-1,300,300,1)

x_train = np.concatenate((x_train,x_augumented))    #concatenate는 파일간에 서로 엮어주는것
y_train = np.concatenate((y_train,y_augumented))    #concatenate는 파일간에 서로 엮어주는것

y_train = pd.get_dummies(y_train)
y_test = pd.get_dummies(y_test)

model = Sequential()
model.add(Conv2D(32,(3,3), input_shape=(300,300,1),strides=1,padding='same'))
model.add(MaxPooling2D())
model.add(Dropout(0.5))
model.add(Conv2D(64,(3,3), activation='relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.5))
model.add(Flatten())
# model.add(Dense(128,activation='relu'))
# model.add(Dense(256,activation='relu'))
# model.add(Dense(512,activation='relu'))
model.add(BatchNormalization())
model.add(Dense(2,activation='relu'))
model.add(Dense(2,activation='softmax'))

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
Loss: 0.512067437171936
Accuracy: 0.7749999761581421
걸린 시간: 49

'''
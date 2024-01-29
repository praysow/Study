from keras.datasets import cifar100
from keras.models import Sequential
from keras.layers import Dense,Conv2D,MaxPooling2D,Dropout,Flatten,BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
(x_train,y_train),(x_test,y_test) = cifar100.load_data()
x_train = x_train/255.
x_test = x_test/255.
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

augumet_size = 20000

randidx= np.random.randint(x_train.shape[0],size=augumet_size)
x_augumented = x_train[randidx].copy()
y_augumented = y_train[randidx].copy()
x_augumented= x_augumented.reshape(-1,32,32,3)


x_augumented = train_datagen.flow(
    x_augumented,
    y_augumented,
    batch_size=augumet_size,
    shuffle=False,
    ).next()[0]

x_train = x_train.reshape(-1,32,32,3)
x_test = x_test.reshape(-1,32,32,3)

x_train = np.concatenate((x_train,x_augumented))    #concatenate는 파일간에 서로 엮어주는것
y_train = np.concatenate((y_train,y_augumented))    #concatenate는 파일간에 서로 엮어주는것


print(x_train.shape,y_train.shape) # (60000, 32, 32, 3) (60000, 1)
print(x_test.shape,y_test.shape)

# y_train = pd.get_dummies(y_train)
# y_test = pd.get_dummies(y_test)
y_train = pd.get_dummies(y_train.flatten())  # y_train을 one-hot 인코딩하기 전에 flatten합니다.
y_test = pd.get_dummies(y_test.flatten())  

model = Sequential()
model.add(Conv2D(32,(3,3), input_shape=(32,32,3),strides=1,padding='same'))
model.add(MaxPooling2D())
model.add(Dropout(0.5))
model.add(Conv2D(64,(3,3), activation='relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dense(256,activation='relu'))
model.add(Dense(512,activation='relu'))
model.add(BatchNormalization())
model.add(Dense(256,activation='relu'))
model.add(Dense(100,activation='softmax'))

from keras.callbacks import EarlyStopping,ModelCheckpoint
es= EarlyStopping(monitor='val_loss',mode='auto',patience=100,verbose=1,restore_best_weights=True)
mcp = ModelCheckpoint(monitor='val_loss',mode='auto',verbose=1,save_best_only=True,
                      filepath='../_data/_save/MCP/keras31-7.hdf5')
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
Loss: 2.3716225624084473
Accuracy: 0.4339999854564667
걸린 시간: 666
'''

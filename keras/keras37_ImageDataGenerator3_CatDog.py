#test 폴더는 사용하지 말것
import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, BatchNormalization, MaxPooling2D,Dropout,AveragePooling2D
from sklearn.model_selection import train_test_split
import time
from sklearn.preprocessing import OneHotEncoder
train_dategen=ImageDataGenerator(
     rescale=1.255,                      #앞에 1.에서 .은 부동소수점 계산하겠다는 것이다
     horizontal_flip=True,           #수평 뒤집기
     vertical_flip=True,             #수직 뒤집기
     width_shift_range=0.1,           #평행이동 0.1=10%이동
     height_shift_range=0.1,          #
     rotation_range=5,               #정해진 각도 만큼 이미지를 회전 (5도 회전)
     zoom_range=1.0,                 #1.2배 확대혹은 축소
     shear_range=0.9,                 #좌표 하나를 고정시키고 다른 몇개의 좌표를 이동시키는 변환
     fill_mode='nearest'
     )

test_datagen = ImageDataGenerator(rescale=1./255)

# path_cat= "c:/_data/image/cat_and_dog/train"
path_dog= "c:/_data/image/cat_and_dog/train"
xy_train=train_dategen.flow_from_directory(path_dog,target_size=(100,100),batch_size=1000,class_mode='binary')         #batch_size을 최대로 하면 x데이터를 통으로 가져올수있다



# print(xy_train)
xy_test=test_datagen.flow_from_directory(path_dog,target_size=(100,100),batch_size=1000,class_mode='binary')
# print(np.unique(xy_train,return_counts=True))

x_train = xy_train[0][0]
y_train = xy_train[0][1]
x_test = xy_test[0][0]
y_test = xy_test[0][1]

x_train,x_test,y_train,y_test=train_test_split(xy_train[0][0],xy_train[0][1],train_size=0.8,random_state=1)

print(x_train.shape,y_train.shape)
print(x_test.shape,y_test.shape)



#2. 모델구성
model = Sequential()
model.add(Conv2D(16, (2, 2), input_shape=(100,100,3)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(12))
model.add(Dense(12))
model.add(Dense(1, activation='sigmoid'))

#3. 모델 컴파일 및 학습
from keras.callbacks import EarlyStopping,ModelCheckpoint
es= EarlyStopping(monitor='val_loss',mode='auto',patience=20,verbose=1,restore_best_weights=True)
mcp = ModelCheckpoint(monitor='val_loss',mode='auto',verbose=1,save_best_only=True,
                      filepath='../_data/_save/MCP/keras31-1.hdf5')
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
start_t = time.time()
model.fit(x_train,y_train,epochs=1000, batch_size=500, verbose=2,
          validation_data=(x_test, y_test),
           callbacks=[es,mcp]
        )
end_t= time.time()

#4. 모델 평가
result = model.evaluate(x_test, y_test)
y_submit= model.predict(x_test)


print("Loss:", result[0])
print("Accuracy:", result[1])
print("걸린 시간:", round(end_t - start_t))

'''
Loss: 76.31924438476562
Accuracy: 0.5299999713897705
걸린 시간: 11

'''
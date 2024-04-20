#test 폴더는 사용하지 말것
import time
import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, BatchNormalization, MaxPooling2D,Dropout,AveragePooling2D
from sklearn.model_selection import train_test_split
import time
from sklearn.preprocessing import OneHotEncoder
start_t = time.time()
train_dategen=ImageDataGenerator(
     rescale=1./255,                      #앞에 1.에서 .은 부동소수점 계산하겠다는 것이다
     # horizontal_flip=True,           #수평 뒤집기
     # vertical_flip=True,             #수직 뒤집기
     # width_shift_range=0.1,           #평행이동 0.1=10%이동
     # height_shift_range=0.1,          #높이 방향만큼 이동
     # rotation_range=5,               #정해진 각도 만큼 이미지를 회전 (5도 회전)
     # zoom_range=1.0,                 #1.2배 확대혹은 축소
     # shear_range=0.9,                 #좌표 하나를 고정시키고 다른 몇개의 좌표를 이동시키는 변환
     # fill_mode='nearest'
     )

batch1=200
batch2=200

test_datagen = ImageDataGenerator(rescale=1./255)

# path_train= "c:/_data/image/cat_and_dog/train/"
# path_test= "c:/_data/image/cat_and_dog/test/"
# xy_train=train_dategen.flow_from_directory(path_train,target_size=(batch1,batch2),batch_size=1600,class_mode='binary',color_mode='grayscale')         #batch_size을 최대로 하면 x데이터를 통으로 가져올수있다


# print(xy_train)
path_test= "c:/_data/image/cat_and_dog/test/"
test=test_datagen.flow_from_directory(path_test,target_size=(batch1,batch2),batch_size=1600,class_mode='binary',color_mode='grayscale')            #원본데이터는 최대한 건드리지 말자,원본데이터는 각각다르니 target_size를 통해서 사이즈를 동일화시킨다
# print(np.unique(xy_train,return_counts=True))


# x = xy_train[0][0]                 #batch가 100일때 100의 0번째이다
# y = xy_train[0][1]                 #batch가 100일때 100의 1번째이다
# x_test = xy_test[0][0]
# y_test = xy_test[0][1]
# print(xy_train[0][0].shape,xy_train[0][1].shape)  #(800, 150, 150, 3) (800,)



# x_train,x_test,y_train,y_test=train_test_split(xy_train[0][0],xy_train[0][1],train_size=0.8,random_state=1,stratify=y)
# x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,random_state=1,stratify=y)

np_path='c:/_data/_save_npy/'
# np.save(np_path + 'keras39_3_x_train.npy', arr=x)
# np.save(np_path + 'keras39_3_y_train.npy', arr=y)
np.save(np_path + 'keras39_3_x_test.npy', arr=test)


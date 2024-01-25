#다중분류만들기

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
train_dategen=ImageDataGenerator(rescale=1/255)

batch1=300
batch2=300

test_datagen = ImageDataGenerator(rescale=1/255)

path_train= "c:/_data/image/horse_human/train/"
# path_test= "c:/_data/image/brain/test/"
xy_train=train_dategen.flow_from_directory(path_train,target_size=(batch1,batch2),batch_size=200,class_mode='binary',color_mode='grayscale')         #batch_size을 최대로 하면 x데이터를 통으로 가져올수있다


# print(xy_train)
xy_test=test_datagen.flow_from_directory(path_train,target_size=(batch1,batch2),batch_size=100,class_mode='binary',color_mode='rgb')            #원본데이터는 최대한 건드리지 말자,원본데이터는 각각다르니 target_size를 통해서 사이즈를 동일화시킨다


x = xy_train[0][0]                 #batch가 100일때 100의 0번째이다
y= xy_train[0][1]                 #batch가 100일때 100의 1번째이다
x_test = xy_test[0][0]
y_test = xy_test[0][1]


print(x.shape,y.shape)  #(200, 300, 300, 1) (200,)
print(x_test.shape,y_test.shape)    #(200, 300, 300, 3) (200,)

np_path='c:/_data/_save_npy/'
# np.save(np_path + 'keras39_7_x_train.npy', arr=x)
# np.save(np_path + 'keras39_7_y_train.npy', arr=y)
np.save(np_path + 'keras39_7_x_test.npy', arr=xy_test)



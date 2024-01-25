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
train_dategen1=ImageDataGenerator(rescale=1.255)
train_dategen2=ImageDataGenerator(rescale=1.255,rotation_range=90)
train_dategen3=ImageDataGenerator(rescale=1.255,rotation_range=180)
train_dategen4=ImageDataGenerator(rescale=1.255,rotation_range=270)



batch1=200
batch2=200

test_datagen = ImageDataGenerator(rescale=1./255)

path_train= "c:/_data/image/cat_and_dog/train/"
# path_test= "c:/_data/image/cat_and_dog/test/"
xy_train=train_dategen.flow_from_directory(path_train,target_size=(batch1,batch2),batch_size=1600,class_mode='binary',color_mode='grayscale')         #batch_size을 최대로 하면 x데이터를 통으로 가져올수있다


# print(xy_train)
path_test= "c:/_data/image/cat_and_dog/test/"
test=test_datagen.flow_from_directory(path_test,target_size=(batch1,batch2),batch_size=1600,class_mode='binary',color_mode='grayscale')            #원본데이터는 최대한 건드리지 말자,원본데이터는 각각다르니 target_size를 통해서 사이즈를 동일화시킨다
# print(np.unique(xy_train,return_counts=True))


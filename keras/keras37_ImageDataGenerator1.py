import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense



train_dategen=ImageDataGenerator(
    rescale=1.255,                      #앞에 1.에서 .은 부동소수점 계산하겠다는 것이다
    horizontal_flip=True,           #수평 뒤집기
    vertical_flip=True,             #수직 뒤집기
    width_shift_range=0.1,           #평행이동 0.1=10%이동
    height_shift_range=0.1,          #
    rotation_range=5,               #정해진 각도 만큼 이미지를 회전 (5도 회전)
    zoom_range=1.2,                 #1.2배 확대혹은 축소
    shear_range=0.7,                 #좌표 하나를 고정시키고 다른 몇개의 좌표를 이동시키는 변환
    fill_mode='nearest'
    )

test_datagen = ImageDataGenerator(rescale=1./255)


path_train= "c:/_data/image/brain/train"
path_test= "c:/_data/image/brain/test"
xy_train=train_dategen.flow_from_directory(path_train,target_size=(200,200),batch_size=160,class_mode='binary',color_mode='grayscale')         #batch_size을 최대로 하면 x데이터를 통으로 가져올수있다,batch란 이미지를 얼만큼 자를것인가
from sklearn.datasets import load_diabetes
daeasets= load_diabetes()


print(xy_train)
xy_test=test_datagen.flow_from_directory(path_test,target_size=(200,200),batch_size=160,class_mode='binary',color_mode='rgb')       #rgb는 red,green,blue
print(xy_train.next())
print(xy_train[0])
# print(xy_train[16])         #에러 이유 : 전체데이터/batch_size =160/10= 16개인데
                            #[16]는 17번째 값을 빼라고 하니 에러가 난다.
print(xy_train[0][0])       #첫번째 배치의 x
print(xy_train[0][1])       #첫번째 배치의 y
print(xy_train[0][0].shape)     #(10, 200, 200, 3)

print(type(xy_train))
print(type(xy_train[0]))
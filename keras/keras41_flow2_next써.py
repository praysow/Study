from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Dense,Conv2D,MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
(x_train,y_train),(x_test,y_test) = fashion_mnist.load_data()


train_datagen=ImageDataGenerator(
     rescale=1./255,                      #앞에 1.에서 .은 부동소수점 계산하겠다는 것이다
    #  horizontal_flip=True,           #수평 뒤집기
    #  vertical_flip=True,             #수직 뒤집기
    #  width_shift_range=0.1,           #평행이동 0.1=10%이동
    #  height_shift_range=0.1,          #높이방향으로 이동
    #  rotation_range=5,               #정해진 각도 만큼 이미지를 회전 (5도 회전)
    #  zoom_range=1.0,                 #1.2배 확대혹은 축소
    #  shear_range=0.9,                 #좌표 하나를 고정시키고 다른 몇개의 좌표를 이동시키는 변환(마름모형태로 만듬)
    #  fill_mode='nearest'
     )

augumet_size = 40000
# print(x_train[0].shape) #(28,28)
# plt.imshow(x_train[0])
# plt.show()

x_data = train_datagen.flow(np.tile(x_train[0].reshape(28*28),augumet_size).reshape(-1,28,28,1),        #-1은 augument_size 이다
                            np.zeros(augumet_size),
                            batch_size=augumet_size,            #augument만큼 tile이 늘어난다
                            shuffle=False
                            ).next()
print(x_data.image.shape)   #튜플형태라서 에러가남, 왜냐하면 flow에서 튜플형태로 반환함
print(x_data[0].shape)  #(100, 28, 28, 1)
print(x_data[1].shape)  #(100,)
print(np.unique(x_data[1],return_counts=True))

print(x_data[0][0].shape)




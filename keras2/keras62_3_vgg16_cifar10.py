#cifar10으로 모델 완성
#1. 성능 비교
#2. 시간 체크       trainable false 와 true
from keras.models import Sequential
from keras.layers import Dense, Flatten
import numpy as np
import tensorflow as tf
print(tf.__version__)   # 2.9.0
tf.random.set_seed(777) # 이쪽이 가중치 초기화에 영향
np.random.seed(777)

from keras.applications import VGG16
from keras.datasets import cifar10
import time

# data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# model
vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(32,32,3))
# vgg16.trainable = True # 가중치를 동결
vgg16.trainable = False# 전이학습
model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(256,activation='relu'))
model.add(Dense(256,activation='relu'))
model.add(Dense(10,activation='softmax'))

# model.summary()
# Total params: 14,914,378
# Trainable params: 199,690
# Non-trainable params: 14,714,688

# compile & fit
st = time.time()
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics='acc')
model.fit(x_train, y_train, batch_size=512, validation_split=0.2, epochs=100)
et = time.time()

# eval & pred
loss = model.evaluate(x_test,y_test)
print(f"time: {et-st:.2f}sec \nacc:  {loss[1]}")
'''
Loss: 0.7987858057022095
Accuracy: 0.7537999749183655
걸린 시간: 82

Loss: 0.6911240816116333
Accuracy: 0.7680000066757202
걸린 시간: 120

time: 544.86sec
acc:  0.7950999736785889
'''
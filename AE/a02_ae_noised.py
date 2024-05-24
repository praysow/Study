import numpy as np
from keras.datasets import mnist,cifar10
from keras.models import Sequential,Model
from keras.layers import Dense,Input
import tensorflow as tf

tf.random.set_seed(512)
np.random.seed(512)

#1.데이터
(x_train, _),(x_test, _) = cifar10.load_data()

print(x_train.shape)
print(x_test.shape)

# x_train = x_train.reshape(60000,28*28).astype('float32')/255.
# x_test = x_test.reshape(10000,28*28).astype('float32')/255.
x_train = x_train.reshape(50000,32*32*3).astype('float32')/255.
x_test = x_test.reshape(10000,32*32*3).astype('float32')/255.
                                            # 평균 0, 표면 0.1인 정규분포
x_train_noised = x_train+np.random.normal(0,0.1, size = x_train.shape).astype('float32')
x_test_noised = x_test+np.random.normal(0,0.1, size = x_test.shape).astype('float32')
print(x_train_noised.shape, x_test_noised.shape)
print(np.max(x_train_noised), np.min(x_train_noised))
print(np.max(x_test_noised), np.min(x_test_noised))
# (50000, 3072) (10000, 3072)
# 1.5142808 -0.5323977

x_train_noised = np.clip(x_train_noised, 0., 1.)
x_test_noised = np.clip(x_test_noised, 0., 1.)

print(np.max(x_train_noised), np.min(x_train_noised))
print(np.max(x_test_noised), np.min(x_test_noised))


#2.모델
input_img = Input(shape=(3072,))
# input_img = Input(shape=(28*28,))
#인코더
encoded = Dense(64,activation='relu')(input_img)
# encoded = Dense(32,activation='relu')(input_img)
# encoded = Dense(1,activation='relu')(input_img)
# encoded = Dense(128,activation='relu')(input_img)
#레이어
x1 = Dense(64,activation='relu')(encoded)
x2 = Dense(128,activation='relu')(x1)
x3 = Dense(256,activation='relu')(x2)
x4 = Dense(512,activation='relu')(x3)
x5 = Dense(256,activation='relu')(x4)
#디코더
# decoded = Dense(32*32*3,activation='linear')(encoded)
decoded = Dense(32*32*3,activation='sigmoid')(x5)
# decoded = Dense(784,activation='relu')(encoded)
# decoded = Dense(784,activation='tanh')(encoded)
autoencoder = Model(input_img,decoded)
# autoencoder.summary()
#3.컴파일
autoencoder.compile(optimizer='adam',loss='mse',metrics=['accuracy'])
# autoencoder.compile(optimizer='adam',loss='binary_crossentropy')
autoencoder.fit(x_train_noised,x_train_noised,epochs=50,verbose=1)

#4.결과추론
# loss = autoencoder.evaluate(x_train,x_train,verbose=0)
pred = autoencoder.predict(x_test_noised)

import matplotlib.pyplot as plt
n = 10
plt.figure(figsize=(20,4))
for i in range(n):
    ax = plt.subplot(2, n, i+1)
    # plt.imshow(x_test[i].reshape(28,28))
    plt.imshow(x_test[i].reshape(32,32,3))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2, n, i+1+n)
    # plt.imshow(pred[i].reshape(28,28))
    plt.imshow(pred[i].reshape(32,32,3))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()

import numpy as np
from keras.datasets import mnist,cifar10
from keras.models import Sequential,Model
from keras.layers import Dense,Input
import tensorflow as tf
import matplotlib.pyplot as plt
import random
tf.random.set_seed(512)
np.random.seed(512)

#1.데이터
(x_train, _),(x_test, _) = mnist.load_data()

print(x_train.shape)
print(x_test.shape)

x_train = x_train.reshape(60000,28*28).astype('float32')/255.
x_test = x_test.reshape(10000,28*28).astype('float32')/255.
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

def autoencoder(hidden_layer_size):
    model = Sequential()
    model.add(Dense(units=hidden_layer_size,input_shape=(28*28,)))
    # model.add(Dense(784,activation='linear'))
    model.add(Dense(784,activation='sigmoid'))
    # model.add(Dense(784,activation='tanh'))
    return model

# hidden_size = 713   #PCA 1.0
hidden_size = 486   #PCA 0.999
# hidden_size = 331   #PCA 0.99
# hidden_size = 154   #PCA 0.95

model = autoencoder(hidden_layer_size=hidden_size)

#3.컴파일
model.compile(optimizer='adam',loss='mse')
# autoencoder.compile(optimizer='adam',loss='binary_crossentropy')
model.fit(x_train_noised,x_train_noised,epochs=10,verbose=1)

#4.결과추론
# loss = autoencoder.evaluate(x_train,x_train,verbose=0)
pred = model.predict(x_test_noised)

import matplotlib.pyplot as plt
n = 10
plt.figure(figsize=(20,4))
for i in range(n):
    ax = plt.subplot(2, n, i+1)
    plt.imshow(x_test[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2, n, i+1+n)
    plt.imshow(pred[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()

fig, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10),
      (ax11, ax12, ax13, ax14, ax15)
      ) = plt.subplots(3,5, figsize=(20,7))
random_images = random.sample(range(pred.shape[0]),5)

for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]):
    ax.imshow(x_test[random_images[i]].reshape(28,28),cmap='gray')
    if i == 0:
        ax.set_ylabel('INPUT',size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

for i,ax in enumerate([ax6, ax7, ax8, ax9, ax10]):
    ax.imshow(x_test_noised[random_images[i]].reshape(28,28),cmap='gray')
    if i == 0:
        ax.set_ylabel('NOISE',size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

for i,ax in enumerate([ax11, ax12, ax13, ax14, ax15]):
    ax.imshow(pred[random_images[i]].reshape(28,28),cmap='gray')
    if i == 0:
        ax.set_ylabel('OUTPUT',size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()
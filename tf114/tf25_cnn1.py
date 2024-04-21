import tensorflow as tf
from tensorflow.keras.datasets import mnist
from sklearn.metrics import accuracy_score
from tensorflow.keras.utils import to_categorical
tf.compat.v1.set_random_seed(6)

(x_train, y_train), (x_test, y_test) = mnist.load_data()

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)

x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255    #127.5도 가능
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255

# 입력 이미지 placeholder 정의
xp = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
yp = tf.placeholder(tf.float32, shape=[None, 10])

w1 = tf.compat.v1.get_variable('w1',shape=[2,2,1,64])
w2 = tf.compat.v1.get_variable('w2',shape=[3,3,64,32])

                                        #커널사이즈,컬러(채널),필터(아웃풋)

L1 = tf.nn.conv2d(xp,w1,strides = [1,2,1,1],padding='VALID')
L2 = tf.nn.conv2d(L1,w2,strides = [1,2,1,1],padding='VALID')

# model.add(conv2d(64,kenel_size=(2,2),stride=(1,1),input_shape=(28,28,1)))

print(w1)   #<tf.Variable 'w1:0' shape=(2, 2, 1, 64) dtype=float32_ref>
print(L1)    #Tensor("Conv2D:0", shape=(?, 14, 27, 64), dtype=float32)

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

w1 = tf.compat.v1.get_variable('w1',shape=[2,2,1,1],initializer=tf.contrib.layers.xavier_initializer())

with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    
    w1_val = sess.run(w1)
    print(w1_val, '\n', w1_val.shape)
# [[[[ 0.20629984]]

#   [[-0.7825234 ]]]


#  [[[-0.21193743]]

#   [[ 0.27944785]]]]
#  (2, 2, 1, 1)
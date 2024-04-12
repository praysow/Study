import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score
tf.compat.v1.set_random_seed(777)

tf.compat.v1.disable_eager_execution()

# t1 -> 그래프연산
# t2 -> 즉시실행 모드
# 즉시 실행모드 끄면 -> 그래프연산으로 바뀜. 그래서 tensor2로도 tensor1을 돌릴 수 있음

#1 data
from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(60000,28,28,1).astype('float32')/255.
x_test = x_test.reshape(10000,28,28,1).astype('float32')/255.

#2 model
x = tf.compat.v1.placeholder(tf.float32, shape = [None, 28,28,1])
y = tf.compat.v1.placeholder(tf.float32, shape = [None, 10])

# Layer1
w1 = tf.compat.v1.get_variable('w1', shape = [2, 2, 1, 32])
                                      # 커널사이즈, 컬러(채널), 필터(아웃풋)
b1 = tf.compat.v1.Variable(tf.compat.v1.zeros([32], name = 'b1'))

L1 = tf.nn.conv2d(x, w1, strides=[1,1,1,1], padding='VALID') # 4차원이라서 [1,1,1,1], 중간 두개가 stride고 앞뒤의 1은 그냥 shape 맞춰주는용. 두칸씩 하려면 [1,2,2,1]. 2번이 가로 3번이 세로
# model.add(Conv2D(32, kernel_size=(2,2), input_shape=(28,28,1), strides=(1,1)))
L1 += b1    # L1 = L1 + b1
L1 = tf.nn.relu(L1)
L1_maxpool = tf.nn.max_pool2d(L1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')  # 4차원이라 앞뒤로 1넣음


# Layer2
w2 = tf.compat.v1.get_variable('w2', shape = [3, 3, 32, 16])
b2 = tf.compat.v1.Variable(tf.compat.v1.zeros([16], name = 'b2'))

L2 = tf.nn.conv2d(L1_maxpool, w2, strides=[1,1,1,1], padding='SAME') 
L2 += b2
L2 = tf.nn.selu(L2)
L2_maxpool = tf.nn.max_pool2d(L2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')

# print(L2)   # (?, 13, 13, 16)
# print(L2_maxpool)   # (?, 6, 6, 16)

# Layer3
w3 = tf.compat.v1.get_variable('w3', shape = [3, 3, 16, 10])
b3 = tf.compat.v1.Variable(tf.compat.v1.zeros([10], name = 'b3'))

L3 = tf.nn.conv2d(L2_maxpool, w3, strides=[1,1,1,1], padding='SAME') 
L3 += b3
L3 = tf.nn.elu(L3)
# L3 = tf.nn.dropout(L3, rate=rate)
# L3_maxpool = tf.nn.max_pool2d(L3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
# print(L3)   # (?, 6, 6, 10)

# Flatten
L_flat = tf.reshape(L3, [-1, 6*6*10])
print("Flatten : ", L_flat)  # Flatten :  Tensor("Reshape:0", shape=(?, 1152), dtype=float32)

# Layer4 DNN
w4 = tf.compat.v1.get_variable('w4', shape=[6*6*10, 10])
b4 = tf.compat.v1.Variable(tf.compat.v1.zeros([10], name='b4'))
L4 = tf.nn.relu(tf.compat.v1.matmul(L_flat, w4) + b4)

# Layer5 DNN
w5 = tf.compat.v1.get_variable('w5', shape=[10,10])
b5 = tf.compat.v1.Variable(tf.compat.v1.zeros([10], name='b5'))
L5 = tf.nn.relu(tf.matmul(L4, w5) + b5)
hypothesis = tf.nn.softmax(L5)

#3 compile
loss = tf.reduce_mean(-tf.reduce_sum(y*tf.math.log(hypothesis + 1e-7 ),axis=1))
train = tf.compat.v1.train.AdamOptimizer(learning_rate=2e-3).minimize(loss)

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

batch_size = 100
total_batch = int(len(x_train) / batch_size)

epochs = 301
for step in range(epochs):
    avg_loss = 0    # loss == cost
    for i in range(total_batch):
        start = i * batch_size
        end = start + batch_size
        
        batch_x, batch_y = x_train[start:end], y_train[start:end]
        feed_dict = {x:batch_x, y:batch_y}
        
        loss_val, _ = sess.run([loss,train], feed_dict=feed_dict)
        
        avg_loss += loss_val / total_batch
        
        
    if step %10 == 0:
        print(step, "loss : ", avg_loss)


pred = sess.run(hypothesis, feed_dict={x:x_test})
# print(pred) 
pred = sess.run(tf.argmax(pred,axis=1))
# print(pred)
y_data = np.argmax(y_test, axis=1)
# print(y_data)

acc = accuracy_score(y_data,pred)
print("acc : ", acc)

sess.close()

# 300 loss :  0.24822466243058555
# acc :  0.8875
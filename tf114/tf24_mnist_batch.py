import tensorflow as tf
from tensorflow.keras.datasets import mnist
from sklearn.metrics import accuracy_score
(x_train,y_train),(x_test,y_test) = mnist.load_data()

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(x_train.shape,x_test.shape)
print(y_train.shape,y_test.shape)

x_train = x_train.reshape(60000,28*28).astype('float32')/255
x_test = x_test.reshape(10000,28*28).astype('float32')/255

xp = tf.compat.v1.placeholder(tf.float32, shape=[None, 784])
yp = tf.compat.v1.placeholder(tf.float32, shape=[None, 10])
dropout_rate = tf.placeholder(tf.float32)

# Weight와 Bias 변수 정의
w1 = tf.compat.v1.get_variable('w1',shape=[784,128],initializer=tf.contrib.layers.xavier_initializer())
w2 = tf.compat.v1.get_variable('w2',shape=[128,64],initializer=tf.contrib.layers.xavier_initializer())
w3 = tf.compat.v1.get_variable('w3',shape=[64,32],initializer=tf.contrib.layers.xavier_initializer())
w4 = tf.compat.v1.get_variable('w4',shape=[32,16],initializer=tf.contrib.layers.xavier_initializer())
w5 = tf.compat.v1.get_variable('w5',shape=[16,10],initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.compat.v1.Variable(tf.compat.v1.zeros([128]),name='bias1')
b2 = tf.compat.v1.Variable(tf.compat.v1.zeros([64]),name='bias2')
b3 = tf.compat.v1.Variable(tf.compat.v1.zeros([32]),name='bias3')
b4 = tf.compat.v1.Variable(tf.compat.v1.zeros([16]),name='bias4')
b5 = tf.compat.v1.Variable(tf.compat.v1.zeros([10]),name='bias5')

layer1 = tf.compat.v1.matmul(xp,w1)+b1
layer1 = tf.nn.dropout(layer1,rate = 0.3)
layer2 = tf.compat.v1.matmul(layer1,w2)+b2
layer2 = tf.nn.relu(layer2)
layer2 = tf.nn.dropout(layer2,rate = 0.3)
layer3 = tf.nn.softmax(tf.compat.v1.matmul(layer2,w3)+b3)
layer4 = tf.compat.v1.matmul(layer3,w4)+b4

hypothesis = tf.nn.softmax(tf.compat.v1.matmul(layer4, w5) + b5)

# 손실 함수 및 최적화 알고리즘 정의
# loss_fn = tf.reduce_mean(-tf.reduce_sum(yp*tf.log(hypothesis),axis=1))   #categorical crossentropy
loss_fn = tf.compat.v1.losses.softmax_cross_entropy(yp,hypothesis)
train = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.75).minimize(loss_fn)

# 세션 시작
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

epochs = 1001
batch_size = 100
total_batch = int(len(x_train) / batch_size)

# # 학습 루프
# for epoch in range(epochs):
#     avg_loss = 0
#     total_batch = int(len(x_train) / batch_size)

#     for i in range(total_batch):
#         batch_x = x_train[i * batch_size:(i + 1) * batch_size]
#         batch_y = y_train[i * batch_size:(i + 1) * batch_size]

#         _, loss_val = sess.run([train, loss_fn], feed_dict={xp: batch_x, yp: batch_y, dropout_rate: 0.5})
#         avg_loss += loss_val / total_batch

#     if epoch % 20 == 0:
#         print(f"Epoch: {epoch}, Loss: {avg_loss}")
# import numpy as np        

# predictions = sess.run(hypothesis, feed_dict={xp: x_test})

# y_pred_argmax = sess.run(tf.math.argmax(predictions,1))
# predictions_binary = (predictions > 0.5).astype(int)
# acc = accuracy_score(y_test, predictions_binary)
# print('acc', acc)
# # x = np.array(x)
# # y_test = np.array(y_test)

# dropout_rate = tf.compat.v1.placeholder(tf.float32)
# correct_prediction = tf.compat.v1.equal(tf.round(hypothesis), yp)
# accuracy = tf.compat.v1.reduce_mean(tf.cast(correct_prediction, tf.float32))
# final_accuracy = sess.run(accuracy, feed_dict={xp: x_test, yp: y_test, dropout_rate: 1.0})
# print("dropout rate 1.0:", final_accuracy)

# sess.close()

'''
acc 0.7284
dropout rate 1.0: 0.95248
'''
avg_loss = 0

for step in range(epochs):
    
    for i in range(total_batch):
        start = i * batch_size
        end = start + batch_size
        
        batch_x,batch_y=x_train[start:end],y_train[start:end]
        feed_dict = {xp:batch_x,yp:batch_y}
        
        loss_val,_,w,b = sess.run([loss_fn,train,w4,b4],feed_dict=feed_dict)
        avg_loss += loss_val / total_batch
        
        if step % 20 == 0:
            print(f"Epoch: {step}, Loss: {avg_loss}")

predictions = sess.run(hypothesis, feed_dict={xp: x_test})

y_pred_argmax = sess.run(tf.math.argmax(predictions,1))
predictions_binary = (predictions > 0.5).astype(int)
acc = accuracy_score(y_test, predictions_binary)
print('acc', acc)
# x = np.array(x)
# y_test = np.array(y_test)

dropout_rate = tf.compat.v1.placeholder(tf.float32)
correct_prediction = tf.compat.v1.equal(tf.round(hypothesis), yp)
accuracy = tf.compat.v1.reduce_mean(tf.cast(correct_prediction, tf.float32))
final_accuracy = sess.run(accuracy, feed_dict={xp: x_test, yp: y_test, dropout_rate: 1.0})
print("dropout rate 1.0:", final_accuracy)

sess.close()

'''
dropout rate 1.0: 0.90022
'''
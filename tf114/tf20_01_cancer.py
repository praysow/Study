from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import tensorflow as tf

#1.data
dataset = load_breast_cancer()
x,y = dataset.data, dataset.target

scaler = MinMaxScaler()
scaler.fit(x)
scaler.transform(x)
x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.9,random_state=6)
# print(x_train.shape,y_train.shape)  #(90, 4) (90,)
xp = tf.compat.v1.placeholder(tf.float32,shape=[None,30])
yp = tf.compat.v1.placeholder(tf.float32,shape=[None,1])

w1 = tf.compat.v1.Variable(tf.compat.v1.random_normal([30,1]),name='weight1')
w2 = tf.compat.v1.Variable(tf.compat.v1.random_normal([1,10]),name='weight2')
w3 = tf.compat.v1.Variable(tf.compat.v1.random_normal([10,20]),name='weight3')
w4 = tf.compat.v1.Variable(tf.compat.v1.random_normal([20,10]),name='weight4')
w5 = tf.compat.v1.Variable(tf.compat.v1.random_normal([10,1]),name='weight5')
b1 = tf.compat.v1.Variable(tf.compat.v1.zeros([1]),name='bias1')
b2 = tf.compat.v1.Variable(tf.compat.v1.zeros([10]),name='bias2')
b3 = tf.compat.v1.Variable(tf.compat.v1.zeros([20]),name='bias3')
b4 = tf.compat.v1.Variable(tf.compat.v1.zeros([10]),name='bias4')
b5 = tf.compat.v1.Variable(tf.compat.v1.zeros([1]),name='bias5')

layer1 = tf.compat.v1.matmul(xp,w1)+b1   #(n,3)
layer2 = tf.compat.v1.matmul(layer1,w2)+b2   #(n,10)
layer3 = tf.compat.v1.sigmoid(tf.compat.v1.matmul(layer2,w3)+b3)   #(n,100)
layer4 = tf.compat.v1.matmul(layer3,w4)+b4   #(n,10)





hypothesis = tf.nn.sigmoid(tf.compat.v1.matmul(layer4, w5) + b5)

loss_fn = -tf.reduce_mean(yp*tf.log(hypothesis)+(1-yp)+tf.log(1-hypothesis))
# optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.1)
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.01)

train = optimizer.minimize(loss_fn)

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

epochs = 101
import numpy as np
y_train = np.reshape(y_train, (-1, 1))

with tf.Session() as sess:
    sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.global_variables_initializer())
    data_set = {xp:x_train,yp:y_train}
    for step in range(epochs):
        _, loss = sess.run([train,loss_fn], feed_dict=data_set)
        if step % 10 == 0:
            print(f"{step}epo | loss:{loss:<30}")


        
    # final_pred = sess.run(hypothesis,feed_dict=data_set)
    # print(final_pred)

predictions = sess.run(hypothesis, feed_dict={xp: x_test})

# y_predict = x_test * w_v
# R-제곱 계산
predictions_binary = (predictions > 0.5).astype(int)
acc = accuracy_score(y_test, predictions_binary)
print('acc', acc)





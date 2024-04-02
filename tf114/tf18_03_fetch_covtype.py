from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import numpy as np
import pandas as pd
x,y = fetch_covtype(return_X_y=True)
scaler = StandardScaler()
scaler.fit(x)
x = scaler.transform(x)
y = pd.get_dummies(y)
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=6)
#(1437, 64) (1437,)
xp = tf.compat.v1.placeholder(tf.float32, shape=[None, 54])
yp = tf.compat.v1.placeholder(tf.float32, shape=[None, 7])

# Weight와 Bias 변수 정의
w = tf.compat.v1.Variable(tf.compat.v1.random_normal([54, 7]), dtype=tf.float32, name='weight')
b = tf.compat.v1.Variable(tf.compat.v1.zeros([7]), dtype=tf.float32, name='bias')

# 2. 모델 구성
hypothesis = tf.nn.softmax(tf.compat.v1.matmul(xp, w) + b)

# 손실 함수 및 최적화 알고리즘 정의
loss_fn = tf.reduce_mean(-tf.reduce_sum(yp*tf.log(hypothesis),axis=1))   #categorical crossentropy

train = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss_fn)

# 세션 시작
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

epochs = 101

# 학습 루프
for step in range(epochs):
    loss_val_, _ = sess.run([loss_fn, train], feed_dict={xp: x_train, yp: y_train})
    if step % 20 == 0:
        print(f"{step}epo | loss:{loss_val_:<30}")
        
# y_pred = sess.run(hypothesis,feed_dict={xp:x_test})
predictions = sess.run(hypothesis, feed_dict={xp: x_test})

predictions_binary = (predictions > 0.5).astype(int)
acc = accuracy_score(y_test, predictions_binary)
print('acc', acc)
sess.close()

'''
acc 0.42924881457449465
'''
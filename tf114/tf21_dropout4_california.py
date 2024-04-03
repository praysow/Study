# #7. load_diabets
# #8. california
# #9. dacon 따릉이
# #10. kaggle bike
import tensorflow as tf
tf.compat.v1.set_random_seed(6)
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

# 1. 데이터 준비 및 전처리
x, y = fetch_california_housing(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=6)

# Placeholder 정의
xp = tf.compat.v1.placeholder(tf.float32, shape=[None, 8])
yp = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

# Weight와 Bias 변수 정의
w1 = tf.compat.v1.Variable(tf.compat.v1.random_normal([8,3]),name='weight1')
w2 = tf.compat.v1.Variable(tf.compat.v1.random_normal([3,10]),name='weight2')
w3 = tf.compat.v1.Variable(tf.compat.v1.random_normal([10,20]),name='weight3')
w4 = tf.compat.v1.Variable(tf.compat.v1.random_normal([20,10]),name='weight4')
w5 = tf.compat.v1.Variable(tf.compat.v1.random_normal([10,1]),name='weight5')
b1 = tf.compat.v1.Variable(tf.compat.v1.zeros([3]),name='bias1')
b2 = tf.compat.v1.Variable(tf.compat.v1.zeros([10]),name='bias2')
b3 = tf.compat.v1.Variable(tf.compat.v1.zeros([20]),name='bias3')
b4 = tf.compat.v1.Variable(tf.compat.v1.zeros([10]),name='bias4')
b5 = tf.compat.v1.Variable(tf.compat.v1.zeros([1]),name='bias5')

layer1 = tf.compat.v1.matmul(xp,w1)+b1
layer2 = tf.compat.v1.matmul(layer1,w2)+b2
layer2 = tf.nn.dropout(layer2,keep_prob = 0.5)
layer3 = tf.compat.v1.matmul(layer2,w3)+b3
layer4 = tf.compat.v1.matmul(layer3,w4)+b4

hypothesis = tf.compat.v1.matmul(layer4, w5) + b5

# 손실 함수 및 최적화 알고리즘 정의
loss_fn = tf.reduce_mean(tf.square(hypothesis - yp))
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.1)
train = optimizer.minimize(loss_fn)

# 세션 시작
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

epochs = 1001
import numpy as np
y_train = np.reshape(y_train, (-1, 1))
# 학습 루프
for step in range(epochs):
    loss_val_, _ = sess.run([loss_fn, train], feed_dict={xp: x_train, yp: y_train})
    if step % 20 == 0:
        print(f"{step}epo | loss:{loss_val_:<30}")

# 세션 종료

from sklearn.metrics import r2_score, mean_absolute_error

# 훈련 루프 이후

# 시험 데이터에 대한 예측 수행
# predictions = sess.run(hypothesis, feed_dict={x: x_test})
predictions = sess.run(hypothesis, feed_dict={xp: x_test})

# y_predict = x_test * w_v
# R-제곱 계산
r2 = r2_score(y_test, predictions)

# MAE 계산
mae = mean_absolute_error(y_test, predictions)

print("R2:", -r2)
print("MAE:", mae)
y_test = np.array(y_test)
dropout_rate = tf.compat.v1.placeholder(tf.float32)
r2 = 1 - tf.reduce_sum(tf.square(yp - hypothesis)) / tf.reduce_sum(tf.square(yp - tf.reduce_mean(yp)))

# 세션을 만들고 R^2 값을 계산합니다.
final_r2 = sess.run(-r2, feed_dict={xp: x_test, yp: y_test.reshape(-1,1), dropout_rate: 1.0})
print("dropout rate 1.0:", final_r2)

sess.close()
'''
R2: 13.309806283968154
MAE: 2.767148383917337
dropout rate 1.0: -14.1092415
'''


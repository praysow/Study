import tensorflow as tf
tf.compat.v1.set_random_seed(5)

x_data = [[1,2],[2,3],[3,1],[4,3],[5,3],[6,2]]
y_data = [[0],[0],[0],[1],[1],[1]]

x = tf.compat.v1.placeholder(tf.float32,shape = [None,2])
y = tf.compat.v1.placeholder(tf.float32,shape = [None,1])

w = tf.compat.v1.Variable(tf.compat.v1.random_normal([2,1],dtype=tf.float32,name='weight'))
b = tf.compat.v1.Variable(tf.compat.v1.zeros([1]),dtype=tf.float32,name='bias')

#2.모델구성
hypothesis = tf.compat.v1.sigmoid(tf.compat.v1.matmul(x,w)+b)

# #3.compile
# loss_fn = tf.reduce_mean(tf.square(hypothesis-y))
loss_fn = -tf.reduce_mean(y*tf.log(hypothesis)+(1-y)*tf.log(1-hypothesis))
# optimiaer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.01)
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1e-5)

train = optimizer.minimize(loss_fn)

# 세션 시작
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

epochs = 101
import numpy as np
y_data = np.reshape(y_data, (-1, 1))
# 학습 루프
for step in range(epochs):
    loss_val_, _ = sess.run([loss_fn, train], feed_dict={x: x_data, y: y_data})
    if step % 20 == 0:
        # print(f"{step}epo | loss:{loss_val_:<30}")
        print(step,"loss:",loss_val_)
        
###sess.run을 통과해서 나오는 데이터는 넘파이의 형태이다
# 세션 종료

from sklearn.metrics import r2_score, mean_absolute_error

# 훈련 루프 이후

# 시험 데이터에 대한 예측 수행
predictions = sess.run(hypothesis, feed_dict={x: x_data})

# y_predict = x_test * w_v
# R-제곱 계산
r2 = r2_score(y_data, predictions)

# MAE 계산
mae = mean_absolute_error(y_data, predictions)

print("R-제곱:", r2)
print("MAE:", mae)

sess.close()


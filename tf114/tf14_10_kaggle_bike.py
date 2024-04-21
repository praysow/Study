import tensorflow as tf
tf.compat.v1.set_random_seed(6)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

# 데이터 준비 및 전처리
path= "c:\_data\kaggle\\bike\\"
train_csv = pd.read_csv(path+"train.csv",index_col=0)
test_csv = pd.read_csv(path+"test.csv",index_col=0)
sampleSubmission_csv = pd.read_csv(path+"sampleSubmission.csv")

x= train_csv.drop(['count','casual','registered'], axis=1)
y= train_csv['count']

x_train,x_test,y_train,y_test=train_test_split(x,y, train_size=0.8, random_state=3)

# Placeholder 정의
xp = tf.compat.v1.placeholder(tf.float32, shape=[None, 8])
yp = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

# Weight와 Bias 변수 정의
w = tf.compat.v1.Variable(tf.compat.v1.random_normal([8, 1]), dtype=tf.float32, name='weight')
b = tf.compat.v1.Variable(tf.compat.v1.zeros([1]), dtype=tf.float32, name='bias')

# 모델 구성
hypothesis = tf.compat.v1.matmul(xp, w) + b

# 손실 함수 및 최적화 알고리즘 정의
loss_fn = tf.reduce_mean(tf.square(hypothesis - yp))
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.001)
train = optimizer.minimize(loss_fn)

# 세션 시작
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

epochs = 10001
y_train = np.expand_dims(y_train, axis=1)

# 학습 루프
for step in range(epochs):
    loss_val_, _ = sess.run([loss_fn, train], feed_dict={xp: x_train, yp: y_train})
    if step % 20 == 0:
        print(f"{step}epo | loss:{loss_val_:<30}")

# 시험 데이터에 대한 예측 수행
predictions = sess.run(hypothesis, feed_dict={xp: test_csv})
y_submit = pd.DataFrame(predictions)

predictions[predictions < 0] = 0
# R-제곱 계산
# r2 = r2_score(y_test, predictions)

# MAE 계산
# mae = mean_absolute_error(y_test, predictions)

# print("R-제곱:", r2)
# print("MAE:", mae)

# y_submit=abs(model.predict(test_csv))

sampleSubmission_csv['count'] = y_submit
print(sampleSubmission_csv)
sampleSubmission_csv.to_csv(path +"sampleSubmission_20.csv", index=False)
# 세션 종료
sess.close()

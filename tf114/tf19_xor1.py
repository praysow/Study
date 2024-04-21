import tensorflow as tf
tf.compat.v1.set_random_seed(6)
#1.data

x_data = [[0,0],[0,1],[1,0],[1,1]]
y_data = [[0],[1],[1],[0]]

x = tf.compat.v1.placeholder(tf.float32,shape = [None,2])
y = tf.compat.v1.placeholder(tf.float32,shape = [None,1])

w = tf.compat.v1.Variable(tf.compat.v1.random_normal([2,1]),name='weight')
b = tf.compat.v1.Variable(tf.compat.v1.zeros([1]),name='bias')

hypothesis = tf.nn.sigmoid(tf.compat.v1.matmul(x, w) + b)

# 손실 함수 및 최적화 알고리즘 정의
# loss_fn = tf.reduce_mean(tf.square(hypothesis - y))
loss_fn = -tf.reduce_mean(y*tf.log(hypothesis)+(1-y)*tf.log(1-hypothesis))

optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-10)
train = optimizer.minimize(loss_fn)

# 세션 시작
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

epochs = 10100

for step in range(epochs):
    loss_val_, _ = sess.run([loss_fn, train], feed_dict={x: x_data, y: y_data})
    if step % 20 == 0:
        print(f"{step}epo | loss:{loss_val_:<30}")
        
        y_pred = sess.run(hypothesis,feed_dict={x:x_data})

from sklearn.metrics import accuracy_score

predictions = sess.run(hypothesis, feed_dict={x: x_data})

# R-제곱 계산
y_pred = sess.run(hypothesis,feed_dict={x:x_data})
predictions = sess.run(hypothesis, feed_dict={x: x_data})

predictions_binary = (predictions > 0.5).astype(int)
acc = accuracy_score(y_data, predictions_binary)
print('acc', acc)
print(y_pred)
print(y_data)
sess.close()
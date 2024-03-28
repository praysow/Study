import tensorflow as tf

tf.compat.v1.set_random_seed(6)

# 데이터
x1_data = [73., 93., 89., 96., 73.]
x2_data = [80., 88., 91., 89., 66.]
x3_data = [75., 93., 90., 100., 70.]
y_data = [152., 185., 180., 196., 142.]

x1 = tf.compat.v1.placeholder(tf.float32,name='x1_data')
x2 = tf.compat.v1.placeholder(tf.float32,name='x2_data')
x3 = tf.compat.v1.placeholder(tf.float32,name='x3_data')
y = tf.compat.v1.placeholder(tf.float32,name='y')

# w = tf.compat.v1.Variable([10], dtype=tf.float32, name='w')
# w1 = tf.compat.v1.Variable([10], dtype=tf.float32, name='weight1')
# w2 = tf.compat.v1.Variable([10], dtype=tf.float32, name='weight2')
# w3 = tf.compat.v1.Variable([10], dtype=tf.float32, name='weight3')
w= tf.compat.v1.Variable(tf.compat.v1.random_normal([1]))
w1= tf.compat.v1.Variable(tf.compat.v1.random_normal([1]),dtype=tf.float32)
w2= tf.compat.v1.Variable(tf.compat.v1.random_normal([1]),dtype=tf.float32)
w3= tf.compat.v1.Variable(tf.compat.v1.random_normal([1]))
b = tf.compat.v1.Variable(0, dtype=tf.float32)

hypothesis = x1_data * w1 + x2_data * w2 + x3_data * w3 + b

# loss_fn = tf.reduce_mean(tf.abs(hypothesis - y_data))  # mae
# optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.001)
# train = optimizer.minimize(loss_fn)

loss_fn = tf.reduce_mean(tf.compat.v1.square(hypothesis-y))
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(loss_fn)

# fit
EPOCHS = 10000
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

for step in range(EPOCHS):
    loss,_ = sess.run([loss_fn,train],feed_dict={x1:x1_data,x2:x2_data,x3:x3_data,y:y_data})
    if step % 20 == 0:
        print(f"{step}epo | loss:{loss:<30}")


# init = tf.compat.v1.global_variables_initializer()
# with tf.compat.v1.Session() as sess:
#     data_set = {x1: x1_data,x2: x2_data,x3: x3_data, y: y_data}
#     loss_hist = []
#     sess.run(init)
#     for step in range(EPOCHS):
#         _, loss = sess.run([train, loss_fn], feed_dict=data_set)
#         loss_hist.append(loss)
#         if step % 100 == 0:
#             print(f"{step}epo | loss:{loss:<30}")

#     final_pred = sess.run(hypothesis, feed_dict=data_set)
#     print(final_pred)

from sklearn.metrics import r2_score, mean_absolute_error

# 훈련 루프 이후

# 시험 데이터에 대한 예측 수행
predictions = sess.run(hypothesis, feed_dict={x1: x1_data,x2: x2_data,x3: x3_data, y: y_data})
# y_predict = x_test * w_v
# R-제곱 계산
r2 = r2_score(y, predictions)

# MAE 계산
mae = mean_absolute_error(y, predictions)

print("R-제곱:", r2)
print("MAE:", mae)


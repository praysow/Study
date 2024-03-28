import tensorflow as tf
import matplotlib.pyplot as plt
tf.set_random_seed(6)

x_train= [1,2,3]
y_train= [1,2,3]
x_test= [4,5,6]
y_test= [4,5,6]

x = tf.compat.v1.placeholder(tf.float32)
y = tf.compat.v1.placeholder(tf.float32)

w = tf.compat.v1.Variable([10],dtype=tf.float32, name='weight')
b = tf.Variable(0,dtype=tf.float32)

hypothesis = x * w


#3-1. compile // model.compile(loss='mse',optimizer='sgd)
loss = tf.reduce_mean(tf.square(hypothesis - y))

###################옵티마이저################
# optimizer = tf.train.AdamOptimizer(learning_rate=0.191)
# train = optimizer.minimize(loss_fn)
lr = 0.01
# gradient = tf.reduce_mean((x*w+b-y)*x)


gradient = tf.reduce_mean((x+w-y)*x)
descent = w-lr*gradient
update = w.assign(descent)
###################옵티마이저################
w_history = []
loss_history = []

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

for step in range(21):
    _, loss_v,w_v = sess.run([update,loss,w], feed_dict={x:x_train,y:y_train})
    print(step, '\t',loss_v,'\t',w_v)
    w_history.append(w_v)
    loss_history.append(loss_v)
        
print('------------------------weight___________________')
print('w',w_history)
print('------------------------loss___________________')
print('loss',loss_history)

plt.plot(w_history,loss_history)
plt.xlabel('weight')
plt.ylabel('loss')
# plt.show()
#########r2, mae만들기
from sklearn.metrics import r2_score, mean_absolute_error

# 훈련 루프 이후

# 시험 데이터에 대한 예측 수행
predictions = sess.run(hypothesis, feed_dict={x: x_test})
# y_predict = x_test * w_v
# R-제곱 계산
r2 = r2_score(y_test, predictions)

# MAE 계산
mae = mean_absolute_error(y_test, predictions)

print("R-제곱:", r2)
print("MAE:", mae)
########tensor1
## R-제곱 계산
total_error = tf.reduce_sum(tf.square(tf.subtract(y, tf.reduce_mean(y))))
unexplained_error = tf.reduce_sum(tf.square(tf.subtract(y, hypothesis)))
r_squared = tf.subtract(1.0, tf.divide(unexplained_error, total_error))

# MAE 계산
mae = tf.reduce_mean(tf.abs(tf.subtract(y, hypothesis)))

# 시험 데이터에 대한 예측 수행
predictions = sess.run(hypothesis, feed_dict={x: x_test})

# R-제곱 및 MAE 값 계산
r2_value, mae_value = sess.run([r_squared, mae], feed_dict={x: x_test, y: y_test})

print("R-제곱:", r2_value)
print("MAE:", mae_value)


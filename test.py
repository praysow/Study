import tensorflow as tf

# 랜덤 시드 설정
tf.compat.v1.set_random_seed(6)

# 데이터
x_data = [1, 2, 3, 4, 5]
y_data = [3, 5, 7, 9, 11]

# 플레이스홀더
_x = tf.placeholder(tf.float32)
_y = tf.placeholder(tf.float32)

# 변수
w = tf.Variable(1.99, dtype=tf.float32)
b = tf.Variable(0.99, dtype=tf.float32)

# 모델
pred = _x * w + b

# 손실 함수 및 최적화
loss_fn = tf.reduce_mean(tf.abs(pred - _y))  # MAE
optimizer = tf.train.AdamOptimizer(learning_rate=0.191)
train = optimizer.minimize(loss_fn)

# 학습
EPOCHS = 101

init = tf.global_variables_initializer()
sess = tf.InteractiveSession()
sess.run(init)
for step in range(EPOCHS):
    _, loss, weight, bias = sess.run([train, loss_fn, w, b], feed_dict={_x: x_data, _y: y_data})
    if step % 1 == 0:
        print(f"{step} 에폭 | 손실: {loss:<30} | 가중치: {weight:<30} | 편향: {bias:<30}")
        
final_pred = sess.run(pred, feed_dict={_x: x_data, _y: y_data})
print(final_pred)

x_pred_data = [6, 7, 8]

# 테스트용 플레이스홀더
x_test = tf.placeholder(tf.float32, shape=[None])

# 예측값
y_predict = x_test * w + b
sess.run(tf.global_variables_initializer())
y_pred_result = y_predict.eval(feed_dict={x_test: x_pred_data})  # eval() 사용
print('[6, 7, 8]의 예측 :', y_pred_result)

sess.close()  # 세션을 닫습니다.

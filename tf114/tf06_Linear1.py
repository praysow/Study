import tensorflow as tf
tf.set_random_seed(777)

#1. 데이터

x = [1,2,3]
y = [1,2,3]

w = tf.Variable(111, dtype=tf.float32)
b = tf.Variable(0, dtype=tf.float32)

#2. 모델구성
# y = wx + b
# hypothesis = w * x + b
hypothesis = x * w + b

#3. 컴파일,훈련
loss = tf.reduce_mean(tf.square(hypothesis-y))      #예측값 - 실제값

# model.compile(loss='mse',optimizer='sgd')
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(loss)

#3-2. 훈련

sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())

# model.fit
epochs = 6000
for step in range(epochs):
    sess.run(train)
    if step % 20 == 0:
        print(step,sess.run(loss),sess.run(w),sess.run(b))  #verbose와 model.weight에서 봤던 애들.
sess.close()


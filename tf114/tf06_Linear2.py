import tensorflow as tf
tf.set_random_seed(777)

#1. 데이터

x = [1,2,3]
y = [1,2,3]

w = tf.Variable(111, dtype=tf.float32)
b = tf.Variable(0, dtype=tf.float32)
#linear
hypothesis = x*w+b
#compile,fit
loss = tf.reduce_mean(hypothesis - y)

#3-1 compile
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(loss)

#3-2 fit
sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())#초기화?

epochs = 100

for step in range(epochs) :
    sess.run(train)
    if step % 20 == 0:
        print(step,sess.run(loss),sess.run(w),sess.run(b))
sess.close()


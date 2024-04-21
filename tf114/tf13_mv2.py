import tensorflow as tf
tf.compat.v1.set_random_seed(6)

# 데이터
x_data = [[73,51,65],
          [92,98,11],
          [89,31,33],
          [99,33,100],
          [17,66,79]]
y_data = [[152],[185],[180],[205],[142]]

x = tf.compat.v1.placeholder(tf.float32,shape = [None,3])
# input_shape=(3,)
y = tf.compat.v1.placeholder(tf.float32,shape = [None,1])
# input_shape=(1,)

w = tf.compat.v1.Variable(tf.compat.v1.random_normal([3,1]),name = 'weight')
b = tf.compat.v1.Variable(0, dtype=tf.float32,name='bias')

# 모델
# hypothesis = x*w + b
hypothesis = tf.compat.v1.matmul(x,w)+b
loss_val = tf.reduce_mean(tf.compat.v1.square(hypothesis-y))
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1e-6)
train = optimizer.minimize(loss_val)

epochs = 1000
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

for step in range(epochs):
    loss_val_, _ = sess.run([loss_val, train], feed_dict={x: x_data, y: y_data})
    if step % 20 == 0:
        print(f"{step}epo | loss:{loss_val_:<30}")

sess.close()
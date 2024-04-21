import tensorflow as tf
tf.set_random_seed(777)

# data
x = [1,2,3,4,5]
y = [3,5,7,9,11]

w = tf.Variable(tf.random_normal([1]),dtype=tf.float32)
b = tf.Variable(0,dtype=tf.float32)

# model
pred = x*w + b

loss_fn = tf.reduce_mean(tf.abs(pred - y))  # mae
optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
train = optimizer.minimize(loss_fn)

# fit
EPOCHS = 30000

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for step in range(EPOCHS):
        sess.run(train)
        if step % 100 == 0:
            print(f"{step}epo | loss:{sess.run(loss_fn):<30} | weight: {sess.run(w)[0]:<30} | bias: {sess.run(b):<30}")
        
    final_pred = sess.run(pred)
    print(final_pred)
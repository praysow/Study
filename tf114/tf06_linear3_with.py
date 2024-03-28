import tensorflow as tf
tf.set_random_seed(777)

# data
x = [1,2,3,4,5]
y = [1,2,3,4,5]

w = tf.Variable(111,dtype=tf.float32)
b = tf.Variable(0,dtype=tf.float32)

# model
pred = x*w + b

# compile
loss_fn = tf.reduce_mean(tf.square(pred-y))
optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
train = optimizer.minimize(loss_fn)

# fit
EPOCHS = 30000

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for step in range(EPOCHS):
        sess.run(train)
        if step % 100 == 0:
            print(f"{step}epo | loss:{sess.run(loss_fn):<30} | weight: {sess.run(w):<30} | bias: {sess.run(b):<30}")
        
    final_pred = sess.run(pred)
    print(final_pred)
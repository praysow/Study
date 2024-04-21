import tensorflow as tf
tf.set_random_seed(777)

# data
x_data = [1,2,3,4,5]
y_data = [3,5,7,9,11]

_x = tf.placeholder(tf.float32)
_y = tf.placeholder(tf.float32)

w = tf.Variable(tf.random_normal([1]),dtype=tf.float32)
b = tf.Variable(0,dtype=tf.float32)

# model
pred = _x*w + b

loss_fn = tf.reduce_mean(tf.abs(pred - _y))  # mae
optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
train = optimizer.minimize(loss_fn)

# fit
EPOCHS = 10000

init = tf.global_variables_initializer()
with tf.Session() as sess:
    data_set = {_x:x_data,_y:y_data}
    sess.run(init)
    best_loss = 987654321
    best_setting = {'w':None,'b':None}
    for step in range(EPOCHS):
        _, loss, weight, bias = sess.run([train,loss_fn,w,b], feed_dict=data_set)
        if loss < best_loss:
            best_loss = loss
            best_setting = {'w':weight,'b':bias}
            print(f"Best: {step}epo's loss={loss}")
        # if step % 100 == 0:
        #     print(f"{step}epo | loss:{loss:<30} | weight: {weight[0]:<30} | bias: {bias[0]:<30}")
        
    w = tf.compat.v1.constant(best_setting['w'])
    b = tf.compat.v1.constant(best_setting['b'])
    new_pred = _x*w + b
    new_loss_fn = tf.reduce_mean(tf.abs(new_pred - _y))  # mae
    print(best_setting)
    print(f"conpare final loss: {sess.run(loss_fn,feed_dict=data_set)}")
    print(f"final loss: {sess.run(new_loss_fn,feed_dict=data_set)}")    # 
    print(f"w: {sess.run(w)}, b: {sess.run(b)}")
    final_pred = sess.run(pred,feed_dict=data_set)
    print(final_pred)
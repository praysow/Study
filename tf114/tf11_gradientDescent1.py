import tensorflow as tf
import matplotlib.pyplot as plt
tf.set_random_seed(6)

x_train= [1,2,3]
y_train= [1,2,3]

x = tf.compat.v1.placeholder(tf.float32)
y = tf.compat.v1.placeholder(tf.float32)

w = tf.compat.v1.Variable([10],dtype=tf.float32, name='weight')

hypothesis = x * w


#3-1. compile // model.compile(loss='mse',optimizer='sgd)
loss = tf.reduce_mean(tf.square(hypothesis - y))

###################옵티마이저################
optimizer = tf.train.AdamOptimizer(learning_rate=0.191)
# train = optimizer.minimize(loss_fn)
lr = 0.01
gradient = tf.reduce_mean((x*w-y)*x)

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
plt.show()
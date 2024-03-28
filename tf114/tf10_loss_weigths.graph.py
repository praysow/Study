import tensorflow as tf
import matplotlib.pyplot as plt
tf.set_random_seed(6)

x= [1,2]
y= [1,2]

w = tf.compat.v1.placeholder(tf.float32)

hypothesis = x * w

loss = tf.reduce_mean(tf.square(hypothesis - y))

w_history = []
loss_history = []

with tf.compat.v1.Session() as sess:
    for i in range(-30,50):
        curr_w = i
        curr_loss = sess.run(loss, feed_dict={w:curr_w})
        
        w_history.append(curr_w)
        loss_history.append(curr_loss)
        
print('------------------------weight___________________')
print(w_history)
print('------------------------loss___________________')
print(loss_history)

plt.plot(w_history,loss_history)
plt.xlabel('weight')
plt.ylabel('loss')
plt.show()
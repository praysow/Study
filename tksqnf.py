import tensorflow as tf
tf.compat.v1.set_random_seed(6)
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_absolute_error

#1. 데이터
x,y = load_diabetes(return_X_y=True)
# print(x.shape,y.shape)  #(442, 10) (442,)

x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.8,random_state=6)



xp = tf.compat.v1.placeholder(tf.float32, shape = [None,10])
yp = tf.compat.v1.placeholder(tf.float32, shape = [None,1])

w = tf.compat.v1.Variable(tf.compat.v1.random_normal([10,1]),dtype=tf.float32,name= 'weight')
b = tf.compat.v1.Variable(tf.compat.v1.zeros([1]),dtype=tf.float32,name='bias')
# b = tf.compat.v1.Variable(0,dtype=tf.float32,name='bias')

#2.모델
hypothesis = tf.compat.v1.matmul(xp,w)+b

loss_fn = tf.reduce_mean(tf.square(hypothesis-y))
optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
train = optimizer.minimize(loss_fn)

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

epochs = 101

for step in range(epochs):
    loss_val_, _ = sess.run([loss_fn, train], feed_dict={xp: x_train, yp: y_train})
    if step % 20 == 0:
        print(f"{step}epo | loss:{loss_val_:<30}")

sess.close()
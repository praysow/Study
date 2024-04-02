from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

#1.data
dataset = load_iris()
x,y = dataset.data, dataset.target

x = x[y !=2]
y = y[y !=2]
scaler = StandardScaler()
scaler.fit(x)
scaler.transform(x)
x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.9,random_state=6)
# print(x_train.shape,y_train.shape)  #(90, 4) (90,)
xp = tf.compat.v1.placeholder(tf.float32,shape=[None,4])
yp = tf.compat.v1.placeholder(tf.float32,shape=[None,1])

w = tf.compat.v1.Variable(tf.random_normal([4,1]),dtype=tf.float32,name = 'weight')
b = tf.compat.v1.Variable(tf.zeros([1]),dtype =tf.float32,name = 'bias')

hypothesis = tf.compat.v1.sigmoid(tf.compat.v1.matmul(xp,w)+b)

loss_fn = -tf.reduce_mean(yp*tf.log(hypothesis)+(1-yp)+tf.log(1-hypothesis))
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(loss_fn)

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

epochs = 101
import numpy as np
y_train = np.reshape(y_train, (-1, 1))

with tf.Session() as sess:
    sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.global_variables_initializer())
    data_set = {xp:x_train,yp:y_train}
    for step in range(epochs):
        _, loss, weight, bias = sess.run([train,loss_fn,w,b], feed_dict=data_set)
        if step % 10 == 0:
            print(f"{step}epo | loss:{loss:<30} | weight: {', '.join(map(str, weight[:,0])):<30} | bias: {bias[0]:<30}")


        
    # final_pred = sess.run(hypothesis,feed_dict=data_set)
    # print(final_pred)

predictions = sess.run(hypothesis, feed_dict={xp: x_test})

# y_predict = x_test * w_v
# R-제곱 계산
predictions_binary = (predictions > 0.5).astype(int)
acc = accuracy_score(y_test, predictions_binary)
print('acc', acc)





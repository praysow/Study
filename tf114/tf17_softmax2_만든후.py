import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score
tf.set_random_seed(6)

#1. 데이터
x_data = [[1,2,1,1],
          [2,1,3,2],
          [3,1,3,4],
          [4,1,5,5],
          [1,7,5,5],
          [1,2,5,6],
          [1,6,6,6],
          [1,7,6,7]]

y_data = [[0,0,1],
          [0,0,1],
          [0,0,1],
          [0,1,0],
          [0,1,0],
          [0,1,0],
          [1,0,0],
          [1,0,0]]

x = tf.compat.v1.placeholder(tf.float32, shape = [None,4])
y = tf.compat.v1.placeholder(tf.float32, shape = [None,3])

w = tf.compat.v1.Variable(tf.random_normal([4,3]),name='weight')
b = tf.compat.v1.Variable(tf.zeros([1,3]),name='bias')

hypothesis = tf.nn.softmax(tf.compat.v1.matmul(x,w)+b)

#3.compile
# loss_fn = tf.reduce_mean(tf.compat.v1.square(hypothesis-y))
loss_fn = tf.reduce_mean(-tf.reduce_sum(y*tf.log(hypothesis),axis=1))   #categorical crossentropy

train = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss_fn)


#3.fit
epochs = 10100

with tf.Session() as sess:
    sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.global_variables_initializer())
    data_set = {x:x_data,y:y_data}
    for step in range(epochs):
        _, loss, weight, bias = sess.run([train,loss_fn,w,b], feed_dict=data_set)
        if step % 10 == 0:
            print(f"{step}epo | loss:{loss:<30} ")
            


        
    # final_pred = sess.run(hypothesis,feed_dict=data_set)
    # print(final_pred)
y_pred = sess.run(hypothesis,feed_dict={x:x_data})
y_predict1 = sess.run(tf.argmax(y_pred,1))
y_predcit2 = np.argmax(y_pred,1)
predictions = sess.run(hypothesis, feed_dict={x: x_data})

predictions_binary = (predictions > 0.5).astype(int)
acc = accuracy_score(y_data, predictions_binary)
print('acc', acc)
print(y_pred)
print(y_predict1)
print(y_predcit2)
y_data = np.argmax(y_data)
print(y_data)
sess.close()


#2.digits
#3.fetch_covtype
#4.dacon_wine
#5. dacon_dechul
#6. kaggle_
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


hypothesis = tf.compat.v1.matmul(x,w)+b

#3.compile
loss_fn = tf.reduce_mean(tf.compat.v1.square(hypothesis-y))
train = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1e-5).minimize(loss_fn)


#3.fit
epochs = 101
import numpy as np
y_train = np.reshape(y, (-1, 1))

with tf.Session() as sess:
    sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.global_variables_initializer())
    data_set = {x:x_data,y:y_data}
    for step in range(epochs):
        _, loss, weight, bias = sess.run([train,loss_fn,w,b], feed_dict=data_set)
        print(weight,bias)
        if step % 10 == 0:
            print(f"{step}epo | loss:{loss:<30} ")


        
    # final_pred = sess.run(hypothesis,feed_dict=data_set)
    # print(final_pred)

predictions = sess.run(hypothesis, feed_dict={x: x_data})

# y_predict = x_test * w_v
# R-제곱 계산
predictions_binary = (predictions > 0.5).astype(int)
acc = accuracy_score(y_data, predictions_binary)
print('acc', acc)
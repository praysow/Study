import tensorflow as tf
tf.compat.v1.set_random_seed(6)
#1.data

x_data = [[0,0],[0,1],[1,0],[1,1]]
y_data = [[0],[1],[1],[0]]

#m02_5와 똑같은 레이어로 구성
#2.model

x = tf.compat.v1.placeholder(tf.float32,shape = [None,2])
y = tf.compat.v1.placeholder(tf.float32,shape = [None,1])

w1 = tf.compat.v1.Variable(tf.compat.v1.random_normal([2,3]),name='weight1')
w2 = tf.compat.v1.Variable(tf.compat.v1.random_normal([3,10]),name='weight2')
w3 = tf.compat.v1.Variable(tf.compat.v1.random_normal([10,20]),name='weight3')
w4 = tf.compat.v1.Variable(tf.compat.v1.random_normal([20,10]),name='weight4')
w5 = tf.compat.v1.Variable(tf.compat.v1.random_normal([10,1]),name='weight5')
b1 = tf.compat.v1.Variable(tf.compat.v1.zeros([3]),name='bias1')
b2 = tf.compat.v1.Variable(tf.compat.v1.zeros([10]),name='bias2')
b3 = tf.compat.v1.Variable(tf.compat.v1.zeros([20]),name='bias3')
b4 = tf.compat.v1.Variable(tf.compat.v1.zeros([10]),name='bias4')
b5 = tf.compat.v1.Variable(tf.compat.v1.zeros([1]),name='bias5')

layer1 = tf.compat.v1.matmul(x,w1)+b1   #(n,3)
layer2 = tf.compat.v1.matmul(layer1,w2)+b2   #(n,10)
layer3 = tf.compat.v1.sigmoid(tf.compat.v1.matmul(layer2,w3)+b3)   #(n,100)
layer4 = tf.compat.v1.matmul(layer3,w4)+b4   #(n,10)





hypothesis = tf.nn.sigmoid(tf.compat.v1.matmul(layer4, w5) + b5)



# 손실 함수 및 최적화 알고리즘 정의
# loss_fn = tf.reduce_mean(tf.square(hypothesis - y))
loss_fn = -tf.reduce_mean(y*tf.log(hypothesis)+(1-y)*tf.log(1-hypothesis))
# optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-5)
# train = optimizer.minimize(loss_fn)

train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss_fn)
# 세션 시작
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

epochs = 10100

predicted = tf.cast(hypothesis > 0.5 ,dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted,y),dtype=tf.float32))

for step in range(epochs):
    loss_val_, _ = sess.run([loss_fn, train], feed_dict={x: x_data, y: y_data})
    if step % 200 == 0:
        print(f"{step}epo | loss:{loss_val_:<30}")
        
        y_pred = sess.run(hypothesis,feed_dict={x:x_data})

hypo,pred,acc = sess.run([hypothesis,predicted,accuracy],feed_dict={x:x_data,y:y_data})
print("hypo",hypo)
print('pred',pred)
print('acc',acc)

from sklearn.metrics import accuracy_score

predictions = sess.run(hypothesis, feed_dict={x: x_data})

# R-제곱 계산
y_pred = sess.run(hypothesis,feed_dict={x:x_data})
predictions = sess.run(hypothesis, feed_dict={x: x_data})

predictions_binary = (predictions > 0.5).astype(int)
acc = accuracy_score(y_data, predictions_binary)
print('acc', acc)
print(y_pred)
sess.close()
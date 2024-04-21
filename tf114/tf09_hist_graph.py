# lr 수정해서 epoch 101qjs dlgkfh wnfdutj
# steop = 100 이하 ,w - 1.99, b = 0.99


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
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.191)
train = optimizer.minimize(loss_fn)

# fit
EPOCHS = 101

loss_val_list=[]
w_val_list = []

init = tf.global_variables_initializer()
with tf.Session() as sess:
    data_set = {_x:x_data,_y:y_data}
    sess.run(init)
    for step in range(EPOCHS):
        _, loss, weight, bias = sess.run([train,loss_fn,w,b], feed_dict=data_set)
        w_val_list.append(weight[0])
        loss_val_list.append(loss)
        
        if step % 1 == 0:
            print(f"{step}epo | loss:{loss:<30} | weight: {weight[0]:<30} | bias: {bias:<30}")
        
    # final_pred = sess.run(pred,feed_dict=data_set)
    # print(final_pred)
    x_pred_data = [6,7,8]

    x_test = tf.compat.v1.placeholder(tf.float32, shape=[None])

    y_predcit = x_pred_data*w + b

    print('[6,7,8]의 예측 :',y_predcit)

    sess = tf.compat.v1.Session()
    sess.run(tf.global_variables_initializer())
    y_predict2 = x_test*w + b
    print('6,7,8의 예측', sess.run(y_predict2,feed_dict={x_test:x_pred_data}))



print(loss_val_list)
print(w_val_list)
import matplotlib.pyplot as plt
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.plot(loss_val_list)
plt.xlabel('epochs')
plt.ylabel('loss')
plt.title('Loss')

    # Weight 그래프
plt.subplot(1, 3, 2)
plt.plot(w_val_list)
plt.xlabel('epochs')
plt.ylabel('weight')
plt.title('Weight')
# Weight vs Loss 그래프
plt.subplot(2, 3, 3)
plt.scatter(w_val_list, loss_val_list)
plt.xlabel('weights')
plt.ylabel('loss')
plt.title('Weight vs Loss')

plt.tight_layout()
plt.show()
'''
 weight: 2.0035932064056396             | bias: 1.0024280548095703dkdlvhs
'''
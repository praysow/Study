import tensorflow as tf
tf.compat.v1.set_random_seed(6)

#07_2를 카피해서 아래를 맹글봐
#1.session()//sess.run(변수)
#2.session()//변수.eval(session=sess)
#3.interactiveSession()//변수.eval()


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
optimizer = tf.train.AdamOptimizer(learning_rate=0.191)
train = optimizer.minimize(loss_fn)

# fit
EPOCHS = 101

# init = tf.global_variables_initializer()
# with tf.Session() as sess:
#     data_set = {_x:x_data,_y:y_data}
#     sess.run(init)
#     for step in range(EPOCHS):
#         _, loss, weight, bias = sess.run([train,loss_fn,w,b], feed_dict=data_set)
#         if step % 1 == 0:
#             print(f"{step}epo | loss:{loss:<30} | weight: {weight[0]:<30} | bias: {bias:<30}")
        
#     final_pred = sess.run(pred,feed_dict=data_set)
#     print(final_pred)
    
# x_pred_data = [6,7,8]

# x_test = tf.compat.v1.placeholder(tf.float32, shape=[None])

# y_predcit = x_pred_data*w + b

# print('[6,7,8]의 예측 :',y_predcit)
# sess = tf.compat.v1.Session()
# sess.run(tf.global_variables_initializer())
# y_predict2 = x_test*w + b
# print('6,7,8의 예측', sess.run(y_predict2,feed_dict={x_test:x_pred_data}))
######################################################2번

init = tf.global_variables_initializer()
with tf.Session() as sess:
    data_set = {_x: x_data, _y: y_data}
    sess.run(init)
    for step in range(EPOCHS):
        _, loss, weight, bias = sess.run([train, loss_fn, w, b], feed_dict=data_set)
        if step % 1 == 0:
            print(f"{step} 에폭 | 손실:{loss:<30} | 가중치: {weight[0]:<30} | 편향: {bias:<30}")
        
    final_pred = sess.run(pred, feed_dict=data_set)
    print(final_pred)
    
x_pred_data = [6, 7, 8]  # 예측할 입력 데이터

x_test = tf.placeholder(tf.float32, shape=[None])  # 테스트용 플레이스홀더

y_predict = x_pred_data * w + b  # 예측값

print('[6, 7, 8]의 예측 :', y_predict)
sess = tf.compat.v1.InteractiveSession()
sess.run(tf.global_variables_initializer())
y_predict2 = x_test * w + b
aaa = sess.eval(feed_dict={x_test: x_pred_data})
print('6, 7, 8의 예측', aaa)

'''
 weight: 2.0035932064056396             | bias: 1.0024280548095703dkdlvhs
'''

#################################################3번
# init = tf.global_variables_initializer()
# sess = tf.InteractiveSession()
# sess.run(init)
# for step in range(EPOCHS):
#     _, loss, weight, bias = sess.run([train, loss_fn, w, b], feed_dict={_x: x_data, _y: y_data})
#     if step % 1 == 0:
#         print(f"{step} 에폭 | 손실: {loss:<30} | 가중치: {weight:<30} | 편향: {bias:<30}")
        
# final_pred = sess.run(pred, feed_dict={_x: x_data, _y: y_data})
# print(final_pred)

# x_pred_data = [6, 7, 8]

# # 테스트용 플레이스홀더
# x_test = tf.placeholder(tf.float32, shape=[None])

# # 예측값
# y_predict = x_test * w + b
# sess.run(tf.global_variables_initializer())
# y_pred_result = y_predict.eval(feed_dict={x_test: x_pred_data})  # eval() 사용
# print('[6, 7, 8]의 예측 :', y_pred_result)

# sess.close() 
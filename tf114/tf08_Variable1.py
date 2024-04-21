import tensorflow as tf
tf.compat.v1.set_random_seed(6)

변수 = tf.compat.v1.Variable(tf.random_normal([2]),name='weight')
print(변수)#<tf.Variable 'weight:0' shape=(2,) dtype=float32_ref>
# 1번
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
aaa= sess.run(변수)
print(aaa)
sess.close()
# 2번
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
bbb = 변수.eval(session = sess) #텐서플로 데이터 형인 '변수'를 파이썬에서 볼수 있게 바꿔준다
print(bbb)      #bbb [-0.34180877 -0.8602732 ]
sess.close()

# 3번

sess = tf.compat.v1.InteractiveSession()
sess.run(tf.compat.v1.global_variables_initializer())
ccc = 변수.eval()
print('ccc',ccc)
sess.close()

# 4번

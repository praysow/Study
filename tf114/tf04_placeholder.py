import tensorflow as tf
print(tf.__version__)
print(tf.executing_eagerly())
tf.compat.v1.disable_eager_execution()
# tf.compat.v1.enable_eager_execution()

# node1 = tf.constant(30.0,tf)
# node2 = tf.constant(4.0)
# node3 = tf.add(node1,node2)
# sess = tf.compat.v1.Session()

a = tf.compat.v1.placeholder(tf.float32)
b = tf.compat.v1.placeholder(tf.float32)
add_node = a+b

sess = tf.compat.v1.Session()

print(sess.run(add_node,feed_dict={a:3,b:4}))
print(sess.run(add_node,feed_dict={a:30,b:4.5}))

add_and_triple = add_node * 3
print(add_and_triple)

print(sess.run(add_and_triple))

######################################################되는거
import tensorflow as tf

# TensorFlow 버전 및 실행 모드 확인
print(tf.__version__)
print(tf.executing_eagerly())

# 즉시 실행 비활성화
tf.compat.v1.disable_eager_execution()

# 플레이스홀더 정의
a = tf.compat.v1.placeholder(tf.float32)
b = tf.compat.v1.placeholder(tf.float32)

# 덧셈 연산 정의
add_node = a + b

# 세션 시작
with tf.compat.v1.Session() as sess:
    # 플레이스홀더에 값을 전달하여 덧셈 연산 실행
    result1 = sess.run(add_node, feed_dict={a: 3, b: 4})
    result2 = sess.run(add_node, feed_dict={a: 30, b: 4.5})
    
    print(result1)
    print(result2)

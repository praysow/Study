import tensorflow as tf
print(tf.__version__)
#컨스턴트,베리어블,클래스월드?
hello = tf.constant("hello world")

sess = tf.Session()

print(sess.run(hello))



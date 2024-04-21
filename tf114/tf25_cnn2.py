import tensorflow as tf
import numpy as np
tf.compat.v1.set_random_seed(6)

#1.data

x_train = np.array([[[[1],[2],[3]],
                     [[4],[5],[6]],
                     [[7],[8],[9]]]])

print(x_train.shape)    #(1, 3, 3, 1)

xp = tf.compat.v1.placeholder(tf.float32,[None,2,2,100])

w = tf.compat.v1.constant([[[[1.]],[[0.]]]
                           [[[1.]],[[0.]]]])
print(w.shape)      #tensor("const:0",shape=(2,2,1,1),dtype=float32)

L1 = tf.nn.conv2d(xp,w,strides=(1,2,2,1))
print(L1)

sess = tf.compat.v1.Session()
output = sess.run(L1,feed_dict={xp:x_train})
print('결과')
print(output)
print('=======')

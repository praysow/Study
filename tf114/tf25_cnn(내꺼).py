import tensorflow as tf
from tensorflow.keras.datasets import mnist
from sklearn.metrics import accuracy_score
from tensorflow.keras.utils import to_categorical
tf.compat.v1.set_random_seed(6)

(x_train, y_train), (x_test, y_test) = mnist.load_data()

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)

x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255    #127.5도 가능
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255

# 입력 이미지 placeholder 정의
xp = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
yp = tf.placeholder(tf.float32, shape=[None, 10])

# Convolutional Layer 1
conv1 = tf.layers.conv2d(inputs=xp, filters=32, kernel_size=[3, 3], padding='same', activation=tf.nn.relu)
# Max Pooling Layer 1
pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

# Convolutional Layer 2
conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=[3, 3], padding='same', activation=tf.nn.relu)
# Max Pooling Layer 2
pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

# Flatten Layer
flat = tf.layers.flatten(pool2)

# Fully Connected Layer
fc = tf.layers.dense(inputs=flat, units=1024, activation=tf.nn.relu)

# Dropout Layer
dropout_rate = tf.placeholder(tf.float32)
drop = tf.nn.dropout(fc, keep_prob=dropout_rate)

# Output Layer
logits = tf.layers.dense(inputs=drop, units=10)

# Softmax를 통한 확률 계산
hypothesis = tf.nn.softmax(logits)

# 손실 함수 및 최적화 알고리즘 정의
loss_fn = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=yp, logits=logits))
train = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss_fn)

# 정확도 계산
correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(yp, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 세션 시작
sess = tf.Session()
sess.run(tf.global_variables_initializer())

epochs = 101
batch_size = 10  # 배치 크기 설정

# 학습 루프
for epoch in range(epochs):
    avg_loss = 0
    total_batch = int(len(x_train) / batch_size)

    for i in range(total_batch):
        batch_x = x_train[i * batch_size:(i + 1) * batch_size]
        batch_y = y_train[i * batch_size:(i + 1) * batch_size]

        _, loss_val = sess.run([train, loss_fn], feed_dict={xp: batch_x, yp: batch_y, dropout_rate: 0.5})
        avg_loss += loss_val / total_batch

    if epoch % 20 == 0:
        print(f"Epoch: {epoch}, Loss: {avg_loss}")

# 정확도 출력
final_accuracy = sess.run(accuracy, feed_dict={xp: x_test, yp: y_test, dropout_rate: 1.0})
print("Test Accuracy:", final_accuracy)

sess.close()

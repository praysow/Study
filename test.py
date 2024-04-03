from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# Load breast cancer dataset
dataset = load_breast_cancer()
x, y = dataset.data, dataset.target

# Normalize input features
scaler = MinMaxScaler()
x = scaler.fit_transform(x)

# Split data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, random_state=6)

# Define placeholders for input and output
xp = tf.placeholder(tf.float32, shape=[None, 30])
yp = tf.placeholder(tf.float32, shape=[None, 1])

# Define variables for weights and biases
w1 = tf.Variable(tf.random_normal([30, 1]), name='weight1')
w2 = tf.Variable(tf.random_normal([1, 10]), name='weight2')
w3 = tf.Variable(tf.random_normal([10, 20]), name='weight3')
w4 = tf.Variable(tf.random_normal([20, 10]), name='weight4')
w5 = tf.Variable(tf.random_normal([10, 1]), name='weight5')
b1 = tf.Variable(tf.zeros([1]), name='bias1')
b2 = tf.Variable(tf.zeros([10]), name='bias2')
b3 = tf.Variable(tf.zeros([20]), name='bias3')
b4 = tf.Variable(tf.zeros([10]), name='bias4')
b5 = tf.Variable(tf.zeros([1]), name='bias5')

# Define the neural network layers
layer1 = tf.matmul(xp, w1) + b1
layer2 = tf.matmul(layer1, w2) + b2
layer2_dropout = tf.nn.dropout(layer2, rate=0.5)
layer3 = tf.nn.sigmoid(tf.matmul(layer2_dropout, w3) + b3)
layer4 = tf.matmul(layer3, w4) + b4
hypothesis = tf.nn.sigmoid(tf.matmul(layer4, w5) + b5)

# Define loss function and optimizer
loss_fn = -tf.reduce_mean(yp * tf.log(hypothesis) + (1 - yp) * tf.log(1 - hypothesis))
optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
train = optimizer.minimize(loss_fn)

# Calculate accuracy
correct_prediction = tf.equal(tf.round(hypothesis), yp)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Training
epochs = 101
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    data_set = {xp: x_train, yp: y_train.reshape(-1, 1)}
    for step in range(epochs):
        _, loss = sess.run([train, loss_fn], feed_dict=data_set)
        if step % 10 == 0:
            print(f"{step}epo | loss:{loss:<30}")

    # Evaluate the model
    test_accuracy = sess.run(accuracy, feed_dict={xp: x_test, yp: y_test.reshape(-1, 1)})
    print("Test Accuracy:", test_accuracy)

    final_accuracy = sess.run(accuracy, feed_dict={xp: x_test, yp: y_test.reshape(-1, 1)})
    print("Final Test Accuracy:", final_accuracy)

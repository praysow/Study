from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
import tensorflow as tf
import numpy as np
import pandas as pd
path= "c:/_data/kaggle/비만/"
train=pd.read_csv(path+"train.csv",index_col=0)
test=pd.read_csv(path+"test.csv",index_col=0)
sample=pd.read_csv(path+"sample_submission.csv")
x= train.drop(['NObeyesdad'],axis=1)
y= train['NObeyesdad']
print('*****************************************',np.unique(y,return_counts=True))
# print(train.shape,test.shape)   #(20758, 17) (13840, 16)    NObeyesdad
# print(x.shape,y.shape)  #(20758, 16) (20758,)
print()
lb = LabelEncoder()

y = pd.get_dummies(y)

# 라벨 인코딩할 열 목록
columns_to_encode = ['Gender','family_history_with_overweight','FAVC','CAEC','SMOKE','SCC','CALC','MTRANS']

# 데이터프레임 x의 열에 대해 라벨 인코딩 수행
for column in columns_to_encode:
    lb.fit(x[column])
    x[column] = lb.transform(x[column])

# 데이터프레임 test_csv의 열에 대해 라벨 인코딩 수행
for column in columns_to_encode:
    lb.fit(test[column])
    test[column] = lb.transform(test[column])

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, random_state=367, stratify=y,shuffle=True)

scaler =StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
test = scaler.transform(test)
xp = tf.compat.v1.placeholder(tf.float32, shape=[None, 16])
yp = tf.compat.v1.placeholder(tf.float32, shape=[None, 7])

# Weight와 Bias 변수 정의
w1 = tf.compat.v1.Variable(tf.compat.v1.random_normal([16,3]),name='weight1')
w2 = tf.compat.v1.Variable(tf.compat.v1.random_normal([3,10]),name='weight2')
w3 = tf.compat.v1.Variable(tf.compat.v1.random_normal([10,20]),name='weight3')
w4 = tf.compat.v1.Variable(tf.compat.v1.random_normal([20,10]),name='weight4')
w5 = tf.compat.v1.Variable(tf.compat.v1.random_normal([10,7]),name='weight5')
b1 = tf.compat.v1.Variable(tf.compat.v1.zeros([3]),name='bias1')
b2 = tf.compat.v1.Variable(tf.compat.v1.zeros([10]),name='bias2')
b3 = tf.compat.v1.Variable(tf.compat.v1.zeros([20]),name='bias3')
b4 = tf.compat.v1.Variable(tf.compat.v1.zeros([10]),name='bias4')
b5 = tf.compat.v1.Variable(tf.compat.v1.zeros([7]),name='bias5')

layer1 = tf.compat.v1.matmul(xp,w1)+b1
layer2 = tf.compat.v1.matmul(layer1,w2)+b2
layer3 = tf.nn.softmax(tf.compat.v1.matmul(layer2,w3)+b3)
layer4 = tf.compat.v1.matmul(layer3,w4)+b4

hypothesis = tf.nn.softmax(tf.compat.v1.matmul(layer4, w5) + b5)

# 손실 함수 및 최적화 알고리즘 정의
loss_fn = tf.reduce_mean(-tf.reduce_sum(yp*tf.log(hypothesis),axis=1))   #categorical crossentropy
train = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss_fn)

# 세션 시작
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

# 세션 시작
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

epochs = 1010

# 학습 루프
for step in range(epochs):
    loss_val_, _ = sess.run([loss_fn, train], feed_dict={xp: x_train, yp: y_train})
    if step % 20 == 0:
        print(f"{step}epo | loss:{loss_val_:<30}")
        
# y_pred = sess.run(hypothesis,feed_dict={xp:x_test})
predictions = sess.run(hypothesis, feed_dict={xp: x_test})

predictions_binary = (predictions > 0.5).astype(int)
acc = accuracy_score(y_test, predictions_binary)
print('acc', acc)
sess.close()

'''
acc 0.6189788053949904
'''


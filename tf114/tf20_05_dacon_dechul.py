from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
import tensorflow as tf
import numpy as np
import pandas as pd
path= "c:\_data\dacon\dechul\\"
train_csv=pd.read_csv(path+"train.csv",index_col=0)
test_csv=pd.read_csv(path+"test.csv",index_col=0)
sample_csv=pd.read_csv(path+"sample_submission.csv")
x= train_csv.drop(['대출등급'],axis=1)
y= train_csv['대출등급']


# print(train_csv,train_csv.shape)        (96294, 14)
# print(test_csv,test_csv.shape)          (64197, 13)
# print(sample_csv,sample_csv.shape)      (64197, 2)
print(np.unique(y,return_counts=True))



y=y.values.reshape(-1,1)

ohe = OneHotEncoder(sparse=False)
ohe = OneHotEncoder()
y_ohe = ohe.fit_transform(y).toarray()

# print(y_ohe,y_ohe.shape)


lb=LabelEncoder()
lb.fit(x['대출기간'])
x['대출기간'] = lb.transform(x['대출기간'])
lb.fit(x['근로기간'])
x['근로기간'] = lb.transform(x['근로기간'])
lb.fit(x['주택소유상태'])
x['주택소유상태'] = lb.transform(x['주택소유상태'])
lb.fit(x['대출목적'])
x['대출목적'] = lb.transform(x['대출목적'])

lb.fit(test_csv['대출기간'])
test_csv['대출기간'] =lb.transform(test_csv['대출기간'])

lb.fit(test_csv['근로기간'])
test_csv['근로기간'] =lb.transform(test_csv['근로기간'])

lb.fit(test_csv['주택소유상태'])
test_csv['주택소유상태'] =lb.transform(test_csv['주택소유상태'])

lb.fit(test_csv['대출목적'])
test_csv['대출목적'] =lb.transform(test_csv['대출목적'])


x_train,x_test,y_train,y_test=train_test_split(x,y_ohe,train_size=0.9,random_state=333 ,
                                               stratify=y_ohe
                                               )
print(x.shape,y.shape)
xp = tf.compat.v1.placeholder(tf.float32, shape=[None, 13])
yp = tf.compat.v1.placeholder(tf.float32, shape=[None, 7])

# Weight와 Bias 변수 정의
w1 = tf.compat.v1.Variable(tf.compat.v1.random_normal([13,3]),name='weight1')
w2 = tf.compat.v1.Variable(tf.compat.v1.random_normal([3,10]),name='weight2')
w3 = tf.compat.v1.Variable(tf.compat.v1.random_normal([10,20]),name='weight3')
w4 = tf.compat.v1.Variable(tf.compat.v1.random_normal([20,10]),name='weight4')
w5 = tf.compat.v1.Variable(tf.compat.v1.random_normal([10,7]),name='weight5')
b1 = tf.compat.v1.Variable(tf.compat.v1.zeros([1]),name='bias1')
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
train = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss_fn)

# 세션 시작
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

epochs = 101

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
acc 0.06593977154724819
'''
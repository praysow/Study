
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense
#1 데이터
x_data = np.array([[0,0],[0,1],[1,0],[1,1]])
y_data = np.array([0,1,1,0])
# print(x_data.shape,y_data.shape)
#2.모델
# model = LinearSVC(C=10)
# model = Perceptron()
model = Sequential()
model.add(Dense(3,input_dim=2))
model.add(Dense(10))
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(1,activation='sigmoid'))
#3.훈련
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['acc'])
model.fit(x_data,y_data,batch_size=1,epochs=1000)
#4.평가예측
# acc = model.score(x_data,y_data)
result = model.evaluate(x_data,y_data)
print("loss",result[0])
print("acc",result[1])

y_pred = np.round(model.predict(x_data))
acc2 = accuracy_score(y_data,y_pred)
print("acc2",acc2)
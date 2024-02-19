import numpy as np
from sklearn.svm import LinearSVC
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
#1 데이터
x_data = np.array([[0,0],[0,1],[1,0],[1,1]])
y_data = np.array([0,0,0,1])
# print(x_data.shape,y_data.shape)
#2.모델
model = LinearSVC()
model = Perceptron()
#3.훈련
model.fit(x_data,y_data)
#4.평가예측
acc = model.score(x_data,y_data)
print(acc)

y_pred = model.predict(x_data)
acc2 = accuracy_score(y_data,y_pred)
print("acc2",acc2)
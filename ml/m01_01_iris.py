import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import LinearSVC

# 1.데이터
datasets = load_iris()
x = datasets.data
y = datasets.target
r = int(np.random.uniform(1, 1000))
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8,
                                                    random_state=r,         #850:acc=1
                                                    stratify=y              #stratify는 분류에서만 사용
                                                    )


from sklearn.svm import LinearSVC
model = LinearSVC(C=100)

model.fit(x_train,y_train)

# 4.결과예측
result = model.score(x_test,y_test)
print("acc :", result)
y_predict = model.predict(x_test)
print(y_predict)
# y_test = np.argmax(y_test,axis=1)                   #열의 값을 맞추기 위해서는 argmax를 통과시킨다
# y_predict= np.argmax(y_predict,axis=1)              #소수점으로 나누어져있는것을 가독성있게 0에서1사이값으로 맞춰준다

# from sklearn.metrics import accuracy_score
# print("acc :", result[0])
# print("acc :",result[1])
# print("random값 :", r)

# 분류모델은 f1스코어 회귀모델은 R2
# SVR
#09 boston
#01_캘리포니아
#11_따릉이
#12_케글바이크
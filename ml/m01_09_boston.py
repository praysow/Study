from sklearn.datasets import load_boston
datasets = load_boston()
x = datasets.data
y = datasets.target

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
import numpy as np

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.9,random_state=100)
from sklearn.svm import LinearSVR
model = LinearSVR(C=100)

model.fit(x_train,y_train)

# 4.결과예측
result = model.score(x_test,y_test)
print("acc :", result)
y_predict = model.predict(x_test)
from sklearn.metrics import r2_score
r2=r2_score(y_test,y_predict)
print("R2 score",r2)



# 로스 : 14.19102668762207          (x,y, train_size=0.9,random_state=100
# R2 score 0.8206877810194941       1,100,1,100,1,100,1,100,1epochs=5000, batch_size=10
#R2 0.62이상
from sklearn.datasets import load_diabetes

#1.데이터
datasets= load_diabetes()
x= datasets.data
y= datasets.target

# print(x.shape) #(442,10)
# print(y.shape) #(442,)

# print(datasets.feature_names)
# print(datasets.DESCR)
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
import numpy as np

x_train,x_test,y_train,y_test= train_test_split(x,y,train_size=0.9, random_state=10)
#2.모델구성
model=Sequential()
model.add(Dense(1,input_dim=10))
model.add(Dense(10))
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(1))
model.add(Dense(10))
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(1))


#3.컴파일 훈련
model.compile(loss='mae',optimizer='adam')
model.fit(x_train,y_train, epochs=3000, batch_size=40)

#4.결과예측
loss=model.evaluate(x_test,y_test)
y_predict=model.predict([x_test])
result=model.predict(x)
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score
r2=r2_score(y_test,y_predict)
print("로스 ,",loss)
print("R2 score :",r2)



'''
로스 , 2202.1640625                 train_size=0.9, random_state=10
R2 score : 0.6353372918979825       1,100,1,100,1,100,1     epochs=2000, batch_size=40

로스 , 2156.254638671875            train_size=0.9, random_state=10
R2 score : 0.6429395322865483       10,100,10,100,10,100,10,100,1    epochs=2000, batch_size=40

로스 , 2123.48046875
R2 score : 0.648366723113752

'''





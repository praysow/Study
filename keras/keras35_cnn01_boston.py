from sklearn.datasets import load_boston
datasets = load_boston()
x = datasets.data
y = datasets.target
# print(x.shape) #(506,13)
# print(y.shape) #(506,)
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense,Conv2D,AveragePooling2D,Dropout,Flatten
import numpy as np
x =x.reshape(x.shape[0],13,1,1)

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.9,random_state=100)

# x_train =x_train.reshape(x_train.shape[0],13,1,1)
# x_test =x_test.reshape(x_test.shape[0],13,1,1)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)



model=Sequential()
model.add(Conv2D(32, (1,1),input_shape=(13,1,1),activation='relu'))
# model.add(AveragePooling2D())
model.add(Dropout(0.1))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(1))
model.add(Dense(100))
model.add(Dense(1))
model.add(Dense(100))
model.add(Dense(1))
model.add(Dense(100))
model.add(Dense(51,activation='softmax'))


#3.컴파일 훈련
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics='accuracy')
model.fit(x_train,y_train, epochs=1000)

#4.결과예측
loss=model.evaluate(x_test,y_test)
y_predict=model.predict([x_test])
print("로스 :",loss)
# print("x 예측값",result)

# import matplotlib.pyplot as plt

# from sklearn.metrics import r2_score
# r2=r2_score(y_test,y_predict)
# print("R2 score",r2)



# 로스 : 14.19102668762207          (x,y, train_size=0.9,random_state=100
# R2 score 0.8206877810194941       1,100,1,100,1,100,1,100,1epochs=5000, batch_size=10
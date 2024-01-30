from sklearn.datasets import load_boston
datasets = load_boston()
x = datasets.data
y = datasets.target
# print(x.shape) #(506,13)
# print(y.shape) #(506,)
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense,LSTM
import numpy as np
x =x.reshape(x.shape[0],13,1)

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.9,random_state=100)

# x_train =x_train.reshape(x_train.shape[0],13,1,1)
# x_test =x_test.reshape(x_test.shape[0],13,1,1)


model=Sequential()
model.add(LSTM(units=32,input_shape=(13,1)))
# model.add(AveragePooling2D())
model.add(Dense(100))
model.add(Dense(1))
model.add(Dense(100))
model.add(Dense(1))
model.add(Dense(100))
model.add(Dense(1))
model.add(Dense(100))
model.add(Dense(1))


#3.컴파일 훈련
model.compile(loss='mse',optimizer='adam')
model.fit(x_train,y_train, epochs=1000)

#4.결과예측
loss=model.evaluate(x_test,y_test)
y_predict=model.predict([x_test])
print("로스 :",loss)

# 로스 : 14.19102668762207          (x,y, train_size=0.9,random_state=100
# R2 score 0.8206877810194941       1,100,1,100,1,100,1,100,1epochs=5000, batch_size=10

# 로스 : 17.524744033813477
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense

#1.데이터

x = np.array(range(1,18))
y = np.array(range(1,18))

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15,random_state=10)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.75,random_state=10)

# print(x_train)[12  9  1 13  5 10  7  8  3  2]
# print(x_test)[ 6  4 15]
# print(x_val)[11 17 14 16]
# print(y_train)
# print(y_test)
# print(y_val)


#2. 모델구성
model=Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(70))
model.add(Dense(100))
model.add(Dense(70))
model.add(Dense(50))
model.add(Dense(1))

#3.컴파일 훈련
model.compile(loss='mse',optimizer='adam')
model.fit(x_train, y_train, epochs=1000, batch_size=1,validation_data=(x_val,y_val))                


#4.결과예측
loss=model.evaluate(x_test,y_test)
result = model.predict([x_test])
print("로스 :", loss)
print("7,5,16예측값 :", result)
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score
r2=r2_score(y_test,result)
print("R2 score",r2)
plt.scatter(x,y)
plt.plot(x, result,color='red')
plt.show()

'''
로스 : 4.547473508864641e-13
7,5,16예측값 : [[ 6.0000005]
 [ 4.0000005]
 [14.999999 ]]
PS C:\Study> 

'''
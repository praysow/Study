from keras.models import Sequential
from keras.layers import Dense
import numpy as np
#1. 데이터
x=np.array([1,2,3,4,5,6,7,8,9,10])
y=np.array([1,2,3,4,6,5,7,8,9,10])

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3,
                                                    shuffle=True,
                                                    random_state=10)        #random_state값을 변경하면 값이 랜덤으로 나오는 값이 계속 바뀌니
print(x_train)     #[ 7  4  2  1  8  5 10]                                                #값을 고정하고 싶다면 random_state값을 고정하자
print(y_train)     #[ 7  4  2  1  8  5 10]
print(x_test)      #[9 3 6]
print(y_test)      #[9 3 6]

#2. 모델구성
model=Sequential()
model.add(Dense(30,input_dim=1))
model.add(Dense(80))
model.add(Dense(100))
model.add(Dense(60))
model.add(Dense(50))
model.add(Dense(30))
model.add(Dense(1))

#3.컴파일, 훈련
model.compile(loss='mse',optimizer='adam')
model.fit(x_train,y_train,epochs=100, batch_size=2)

#4.결과예측
loss=model.evaluate(x,y)
result=model.predict([x])
print("로스: ",loss)
print("예측값 :",result)
'''
로스:  0.20141001045703888
예측값 : [[490.02762]]            30,80,100,60,50,30,1,     epochs=100, batch_size=2

'''

import matplotlib.pyplot as plt

plt.scatter(x,y)
plt.plot(x, result,color='red')
plt.show()









import numpy as np
from keras.models import Sequential
from keras.layers import Dense
#1.데이터
#1~10프레임 #11~13validation
# x= np.array([1,17])
# y= np.array([1,17])

x_train=np.array(range(1,11))
y_train=np.array(range(1,11))

x_val=np.array(range(12,14))
y_val=np.array(range(12,14))
# y_val=np.array([x_train,y_train])

x_test=np.array(range(15,18))
y_test=np.array(range(15,18))

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
model.fit(x_train, y_train, epochs=10, batch_size=1,validation_data=(x_val,y_val))                


#4.결과예측
loss=model.evaluate(x_test,y_test)
result=model.predict([x_test])
print("로스 :", loss)
print("예측값 :", result)
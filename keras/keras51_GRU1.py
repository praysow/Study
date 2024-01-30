#Bidirection은 보조적인 역할이다
# 훈련부분에서 갔다가 다시오는것
import numpy as np
from keras.models import Sequential
from keras.layers import Dense,SimpleRNN,Bidirectional,GRU
from sklearn.model_selection import train_test_split
#1. 데이터

datasets = np.array([1,2,3,4,5,6,7,8,9,10])
#타임스텝스 3으로 자를것이다

x=np.array([[1,2,3],
           [2,3,4],
           [3,4,5],
           [4,5,6],
           [5,6,7],
           [6,7,8],
           [7,8,9]])
y=np.array([4,5,6,7,8,9,10])
# print(x.shape,y.shape)  (7, 3) (7,)
x= x.reshape(7,3,1)
#2.모델구성
model=Sequential()
model.add(GRU(units=10,input_shape=(3,1)))
model.add(Dense(50))
model.add(Dense(100))
model.add(Dense(80))
model.add(Dense(140))
model.add(Dense(90))
model.add(Dense(100))
model.add(Dense(120))
model.add(Dense(60))
model.add(Dense(1))

model.summary()
#.컴파일 훈련
# from keras.callbacks import EarlyStopping,ModelCheckpoint
# es= EarlyStopping(monitor='loss',mode='auto',patience=100,verbose=3,restore_best_weights=True)
# model.compile(loss='mse',optimizer='adam')
# model.fit(x,y,epochs=10000,
#           callbacks=[es]
#           )

# #4.결과예측
# result = model.evaluate(x,y)
# print(("loss",result))
# y_pred= np.array([8,9,10]).reshape(1,3,1)
# y_pred=model.predict(y_pred)
# print("8,9,10결과",y_pred)

'''
8,9,10결과 [[10.926798]]

8,9,10결과 [[10.931554]]
'''


import numpy as np
from keras.models import Sequential
from keras.layers import Dense,LSTM,Conv1D,Flatten
from sklearn.model_selection import train_test_split
x=np.array([[1,2,3],[2,3,4],[3,4,5],
            [4,5,6],[5,6,7],[6,7,8],
            [7,8,9],[8,9,10],[9,10,11],
            [10,11,12],[20,30,40],[30,40,50],[40,50,60]])
y=np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])
# y_predict = np.array([50,60,70])

x=x.reshape(-1,3,1,1)

x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.7,random_state=3)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

model=Sequential()
# model.add(LSTM(units=100,input_shape=(3,1)))
model.add(Conv1D(filters=100,kernel_size=2,input_shape=(3,1)))
model.add(Flatten())
model.add(Dense(10))
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(100))
model.add(Dense(1))



#.컴파일 훈련
from keras.callbacks import EarlyStopping,ModelCheckpoint
es= EarlyStopping(monitor='loss',mode='auto',patience=500,verbose=3,restore_best_weights=True)
mcp = ModelCheckpoint(
    monitor='loss',
    mode='auto',
    verbose=1,
    save_best_only=True,
    filepath='..\_data\_save\MCP\지우기.hdf5'
    )
model.compile(loss='mse',optimizer='adam')
model.fit(x,y,epochs=10000,verbose=3,batch_size=1,
          callbacks=[es,mcp]
          )

#4.결과예측
result = model.evaluate(x,y)
y_predict = np.array([50,60,70]).reshape(1,3,1)
y_pred=model.predict(y_predict)
print(("loss",result))
print("50,60,70결과",y_pred)

'''
('loss', 0.00012596952728927135)
50,60,70결과 [[79.533775]]

('loss', 2.236633918073494e-07)
50,60,70결과 [[80.00077]]
'''
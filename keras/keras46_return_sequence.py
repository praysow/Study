import numpy as np
from keras.models import Sequential
from keras.layers import Dense,LSTM,SimpleRNN

x=np.array([[1,2,3],[2,3,4],[3,4,5],
            [4,5,6],[5,6,7],[6,7,8],
            [7,8,9],[8,9,10],[9,10,11],
            [10,11,12],[20,30,40],[30,40,50],[40,50,60]])
y=np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])
# y_predict = np.array([50,60,70])
x=x.reshape(-1,3,1)

model=Sequential()
# model.add(SimpleRNN(units=100,input_shape=(3,1)))
model.add(LSTM(units=100,return_sequences=True,input_shape=(3,1)))
model.add(SimpleRNN(10))
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

# model.summary()

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
x_predict = np.array([50,60,70]).reshape(1,3,1)
y_pred=model.predict(x_predict)
print(("loss",result))
print("50,60,70결과",y_pred)

'''
('loss', 0.00012596952728927135)
50,60,70결과 [[79.533775]]

('loss', 0.00043098529567942023)
50,60,70결과 [[74.66332]]
'''
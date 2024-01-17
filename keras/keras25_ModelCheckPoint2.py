from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model
from keras.layers import Dense
import numpy as np
#현재 사이킷런 버전1.3.0 보스턴 안됨, 그래서 삭제
#pip uninstall scikit-learn
#pip uninstall scikit-image
#pip uninstall scikit-learn-intelex
#pip install scikit-learn==0.23.2
datasets = load_boston()
x = datasets.data
y = datasets.target

# print(x.shape,y.shape)  (506, 13) (506,)

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.9,random_state=100)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# print(np.min(x_train))
# print(np.min(x_test))
# print(np.max(x_train))
# print(np.max(x_test))

model= load_model('..\_data\_save\MCP\keras25_MCP1.hdf5')




'''
model=Sequential()
model.add(Dense(1,input_shape=(13,)))
model.add(Dense(100))
model.add(Dense(1))
model.add(Dense(100))
model.add(Dense(1))
model.add(Dense(100))
model.add(Dense(1))
model.add(Dense(100))
model.add(Dense(1))

#3.컴파일 훈련
from keras.callbacks import EarlyStopping, ModelCheckpoint
es= EarlyStopping(monitor='val_loss',mode='min',patience=100,verbose=1,restore_best_weights=True)
mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    verbose=1,
    save_best_only=True,
    filepath='..\_data\_save\MCP\keras25_MCP1.hdf5'
    )


model.compile(loss='mse',optimizer='adam')
hist= model.fit(x_train, y_train, epochs=1000,batch_size=1000, validation_split=0.1,verbose=2, callbacks=[es,mcp])

#4.결과예측
loss=model.evaluate(x_test,y_test)
y_predict=model.predict([x_test])
result=model.predict(x)
print("로스 :",loss)
# print("x 예측값",result)

import matplotlib.pyplot as plt

from sklearn.metrics import r2_score
r2=r2_score(y_test,y_predict)
print("R2 score",r2)
print("==================================")
print(hist.history)

f1 0.8834007722701109
로스: 0.26390382647514343
acc 0.9125648736953735
r1 7
r2 44
r3 30
r4 24
r5 2
r6 48

'''
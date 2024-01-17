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
    filepath='..\_data\_save\MCP\keras25_MCP3.hdf5'
    )


model.compile(loss='mse',optimizer='adam')
hist= model.fit(x_train, y_train, epochs=1000,batch_size=1000, validation_split=0.1,verbose=2, callbacks=[es,mcp])


model.save('..\_data\_save\keras25_3_save_model.hdf5')

print("==================1. 기본 ================")

#4.결과예측
loss=model.evaluate(x_test,y_test,verbose=0)
y_predict=model.predict(x_test,verbose=0)
result=model.predict(x)
print("로스 :",loss)
# print("x 예측값",result)


from sklearn.metrics import r2_score
r2=r2_score(y_test,y_predict)
print("R2 score",r2)
print("==================2. load_model ================")
model2 = load_model('..\_data\_save\keras25_3_save_model.hdf5')
loss2=model.evaluate(x_test,y_test,verbose=0)
y_predict2=model2.predict(x_test,verbose=0)
result=model.predict(x)
print("로스 :",loss)
# print("x 예측값",result)


from sklearn.metrics import r2_score
r2=r2_score(y_test,y_predict)
print("R2 score",r2)

print("==================3. load_model ================")
model3= load_model('..\_data\_save\MCP\keras25_MCP1.hdf5')
loss3=model.evaluate(x_test,y_test,verbose=0)
y_predict3=model3.predict(x_test,verbose=0)
result=model.predict(x)
print("로스 :",loss)
# print("x 예측값",result)


from sklearn.metrics import r2_score
r2=r2_score(y_test,y_predict)
print("R2 score",r2)

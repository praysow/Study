#restore_best_weights
#save_best_only
#에 대한 고찰
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model
from keras.layers import Dense
from sklearn.metrics import r2_score
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

# model= load_model('..\_data\_save\MCP\keras25_MCP1.hdf5')


# model=Sequential()
# model.add(Dense(1,input_shape=(13,)))
# model.add(Dense(100))
# model.add(Dense(1))
# model.add(Dense(100))
# model.add(Dense(1))
# model.add(Dense(100))
# model.add(Dense(1))
# model.add(Dense(100))
# model.add(Dense(1))

model= load_model('..\_data\_save\MCP\k25\k25_01-17_11-48_0056-63.6249.hdf5')

# #3.컴파일 훈련
# from keras.callbacks import EarlyStopping, ModelCheckpoint
# import datetime
# date= datetime.datetime.now()
# print(date)     #10:52:52.279279
# date = date.strftime("%m-%d_%H-%M")
# print(date) 
# print(type(date))       #<class 'str'> 열 데이터



# path='..//_data//_save//MCP/k25/'
# filename= "{epoch:04d}-{val_loss:.4f}.hdf5"                #epoch:04d는 4자리 숫자까지 표현 val_loss:.4f는 소수4자리까지 표현
# filepath = "".join([path,'k25_',date,'_',filename])            #join의 의미 :filepath라는 공간을 만들어 path,date,filename을 서로 연결해 주세요

# es= EarlyStopping(monitor='val_loss',mode='min',patience=10,verbose=1,restore_best_weights=True)
# mcp = ModelCheckpoint(
#     monitor='val_loss',
#     mode='auto',
#     verbose=1,
#     save_best_only=True,
#     filepath=filepath
#     )


# model.compile(loss='mse',optimizer='adam')
# hist= model.fit(x_train, y_train, epochs=1000,batch_size=1000, validation_split=0.1,verbose=2, callbacks=[es,mcp])




#4.결과예측
loss=model.evaluate(x_test,y_test,verbose=0)
y_predict=model.predict(x_test,verbose=0)
result=model.predict(x)
r2=r2_score(y_test,y_predict)
print("R2 score",r2)
print("로스 :",loss)


#restore_best_weights
#save_best_only
#True, True     R2 score 0.010760000953027427
                #로스 : 78.28987884521484
#True, FalseR2 score 0.19254995566007704
#              로스 : 63.90275573730469
#False, TrueR2 score -0.3303490463183649
               #로스 : 105.28572845458984
#False, False   R2 score -0.8165784608691153
                #로스 : 143.7666473388672


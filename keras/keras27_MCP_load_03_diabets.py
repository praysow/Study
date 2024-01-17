#R2 0.62이상
from sklearn.datasets import load_diabetes

#1.데이터
datasets= load_diabetes()
x= datasets.data
y= datasets.target

# print(x.shape) #(442,10)
# print(y.shape) #(442,)

# print(datasets.feature_names)
# print(datasets.DESCR)
from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model
from keras.layers import Dense
import numpy as np

x_train,x_test,y_train,y_test= train_test_split(x,y,train_size=0.9, random_state=10)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
scaler = MaxAbsScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2.모델구성
# model=Sequential()
# model.add(Dense(1,input_dim=10))
# model.add(Dense(10))
# model.add(Dense(100))
# model.add(Dense(10))
# model.add(Dense(1))
# model.add(Dense(10))
# model.add(Dense(100))
# model.add(Dense(10))
# model.add(Dense(1))


# #3.컴파일 훈련
# from keras.callbacks import EarlyStopping, ModelCheckpoint
# mcp = ModelCheckpoint(
#     monitor='val_loss',
#     mode='auto',
#     verbose=1,
#     save_best_only=True,
#     filepath='..\_data\_save\MCP\keras26_MCP_diabets.hdf5'
#     )
# es= EarlyStopping(monitor='val_loss',mode='min',patience=100,verbose=1,restore_best_weights=True)
# model.compile(loss='mse',optimizer='adam')
# hist= model.fit(x_train, y_train, epochs=10000,batch_size=1000, validation_split=0.1,verbose=2,
#           callbacks=[es,mcp])

# model.save("c:\_data\_save\diabets_1.h5")

model = load_model("c:\_data\_save\MCP\k26\\03_diabets_01-17_14-22_0159-4207.8745.hdf5")


model.summary()

#4.결과예측
loss=model.evaluate(x_test,y_test)
y_predict=model.predict([x_test])
result=model.predict(x)
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score
r2=r2_score(y_test,y_predict)
print("로스 ,",loss)
print("R2 score :",r2)

'''
로스 , 4241.7373046875
R2 score : 0.29759848907230024      save

로스 , 4241.7373046875
R2 score : 0.29759848907230024      load
'''
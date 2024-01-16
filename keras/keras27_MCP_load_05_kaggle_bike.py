from sklearn.metrics import r2_score, mean_squared_error, mean_squared_log_error
from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model
from keras.layers import Dense
import numpy as np
import pandas as pd

#1. 데이터

path= "c:\_data\kaggle\\bike\\"
train_csv = pd.read_csv(path+"train.csv",index_col=0)
test_csv = pd.read_csv(path+"test.csv",index_col=0)
sampleSubmission_csv = pd.read_csv(path+"sampleSubmission.csv")

x= train_csv.drop(['count','casual','registered'], axis=1)
y= train_csv['count']

x_train,x_test,y_train,y_test=train_test_split(x,y, train_size=0.8, random_state=3)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
scaler = MaxAbsScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# #2.모델구성
# model=Sequential()
# model.add(Dense(10,input_dim=8,activation='relu'))
# model.add(Dense(10,activation='relu'))
# model.add(Dense(1,activation='relu'))
# model.add(Dense(20,activation='relu'))
# model.add(Dense(50,activation='relu'))
# model.add(Dense(200,activation='relu'))
# model.add(Dense(50,activation='relu'))
# model.add(Dense(20,activation='relu'))
# model.add(Dense(1,activation='relu'))
# model.add(Dense(10,activation='relu'))
# model.add(Dense(10,activation='relu'))
# model.add(Dense(1))


# #3.컴파일 훈련
# from keras.callbacks import EarlyStopping, ModelCheckpoint
# mcp = ModelCheckpoint(
#     monitor='val_loss',
#     mode='auto',
#     verbose=1,
#     save_best_only=True,
#     filepath='..\_data\_save\MCP\keras26_MCP_kaggle_bike.hdf5'
#     )
# es= EarlyStopping(monitor='val_loss',mode='min',patience=100,verbose=1,restore_best_weights=True)
# model.compile(loss='mse',optimizer='adam')
# hist= model.fit(x_train, y_train, epochs=10000,batch_size=1000, validation_split=0.1,verbose=2,
#           callbacks=[es,mcp])

# model.save("c:\_data\_save\kaggle_bike_1.h5")

model = load_model("c:\_data\_save\\kaagle_bike_1.h5")

model.summary()


#4.결과예측
loss=model.evaluate(x_test,y_test)
y_submit=model.predict(test_csv)
# y_submit=abs(model.predict(test_csv))

sampleSubmission_csv['count'] = y_submit
print(sampleSubmission_csv)
sampleSubmission_csv.to_csv(path +"sampleSubmission_13.csv", index=False)
print("로스 :",loss)
print("음수 개수:",sampleSubmission_csv[sampleSubmission_csv['count']<0].count())

y_predict= model.predict(x_test)

# def RMSE(y_test, y_predict):
#     np.sqrt(mean_squared_error(y_test,y_predict))
#     return np.sqrt(mean_squared_error(y_test,y_predict))
# rmse=RMSE(y_test,y_predict)
# print("RMSE :",rmse)

def RMSLE(y_test, y_predict):
    # np.sqrt(mean_squared_error(y_test,y_predict))
    return np.sqrt(mean_squared_log_error(y_test,y_predict))
rmsle=RMSLE(y_test,y_predict)
print("RMSLE :", rmsle)

'''
로스 : 32414.482421875
음수 개수: datetime    0
count       0
dtype: int64
69/69 [==============================] - 0s 524us/step
RMSLE : 1.5823909639816276


'''
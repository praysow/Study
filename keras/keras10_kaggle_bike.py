from sklearn.metrics import r2_score, mean_squared_error, mean_squared_log_error
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import pandas as pd

#1. 데이터

path= "c:\_data\kaggle\\bike\\"
train_csv = pd.read_csv(path+"train.csv",index_col=0)
test_csv = pd.read_csv(path+"test.csv",index_col=0)
sampleSubmission_csv = pd.read_csv(path+"sampleSubmission.csv")

# print(train_csv)
# print(test_csv)
# print(submission_csv)

# print("train",train_csv.shape)      #(10886, 11)
# print("test",test_csv.shape)       #(6493, 8)
# print("sub",sampleSubmission_csv.shape) #(6493, 2)

#train_csv=train_csv.dropna()
# train_csv=train_csv.fillna(train_csv.mean())
# train_csv=train_csv.fillna(0)
# test_csv=test_csv.fillna(test_csv.mean())
#test_csv=test_csv.fillna(0)

x= train_csv.drop(['count','casual','registered'], axis=1)
y= train_csv['count']

x_train,x_test,y_train,y_test=train_test_split(x,y, train_size=0.8, random_state=3)

#2.모델구성
model=Sequential()
model.add(Dense(10,input_dim=8,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(1,activation='relu'))
model.add(Dense(20,activation='relu'))
model.add(Dense(50,activation='relu'))
model.add(Dense(200,activation='relu'))
model.add(Dense(50,activation='relu'))
model.add(Dense(20,activation='relu'))
model.add(Dense(1,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(1))


#3.컴파일 훈련
model.compile(loss='mse',optimizer='adam')
model.fit(x_train,y_train, epochs=500, batch_size=64, verbose=2)

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
로스 : 22757.763671875
음수 개수: datetime    0
count       0
dtype: int64
69/69 [==============================] - 0s 331us/step
RMSLE : 1.2888789343348863
'''

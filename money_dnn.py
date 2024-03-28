import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import Adam
import joblib

# 데이터 불러오기
path = "c:/_data/dacon/soduc/"
train = pd.read_csv(path+'train.csv', index_col=0)
test = pd.read_csv(path+'test.csv', index_col=0)
sample = pd.read_csv(path+'sample_submission.csv')

# 피처와 타겟 분리
x = train.drop(['Income','Gains','Losses','Dividends'], axis=1)
y = train['Income']
test = test.drop(['Gains','Losses','Dividends'], axis=1)
lb = LabelEncoder()

# 라벨 인코딩할 열 목록
columns_to_encode = ['Gender','Education_Status','Employment_Status','Industry_Status','Occupation_Status','Race','Hispanic_Origin','Martial_Status','Household_Status','Household_Summary','Citizenship','Birth_Country','Birth_Country (Father)','Birth_Country (Mother)','Tax_Status','Income_Status']

# 데이터프레임 x의 열에 대해 라벨 인코딩 수행
for column in columns_to_encode:
    lb.fit(x[column])
    x[column] = lb.transform(x[column])

# 데이터프레임 test_csv의 열에 대해 라벨 인코딩 수행
for column in columns_to_encode:
    lb.fit(test[column])
    test[column] = lb.transform(test[column])

# 데이터 스케일링
scaler = StandardScaler()
x = scaler.fit_transform(x)
test = scaler.transform(test)

x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.9,random_state=8)

#2.모델구성
model = Sequential()
model.add(Dense(500,input_shape=(18,)))
model.add(Dense(1000,activation='swish'))
model.add(BatchNormalization())
model.add(Dense(1500,activation='swish'))
model.add(Dense(2000,activation='swish'))
model.add(BatchNormalization())
model.add(Dense(1500,activation='swish'))
model.add(Dense(1000,activation='swish'))
model.add(BatchNormalization())
model.add(Dense(800,activation='swish'))
model.add(Dense(600,activation='swish'))
model.add(Dense(400,activation='swish'))
model.add(Dense(500,activation='swish'))
model.add(Dense(1))
model.save( "c:/_data/dacon/soduc/")
#3.컴파일,훈련
from keras.callbacks import EarlyStopping,ModelCheckpoint,ReduceLROnPlateau
path='c://_data//_save//MCP/'
filename= "{epoch:04d}-{val_loss:.4f}.hdf5"                #epoch:04d는 4자리 숫자까지 표현 val_loss:.4f는 소수4자리까지 표현
filepath = "".join([path,filename]) 

es= EarlyStopping(monitor='val_loss',mode='min',patience=1000,verbose=1,restore_best_weights=False)
mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    verbose=1,
    save_best_only=True,
    filepath=filepath,
    period = 5
    )
rlr = ReduceLROnPlateau(monitor='val_loss',patience=1,mode='auto',verbose=1,factor=0.5)
lr = 0.001
model.compile(loss='mse',optimizer=Adam(learning_rate=lr))
hist= model.fit(x_train, y_train, epochs=10000,batch_size=3000, validation_split=0.1,verbose=2, callbacks=[es,rlr])

#4.결과
loss = model.evaluate(x_test,y_test)
y_pred = model.predict(x_test)
rmse = mean_squared_error(y_test, y_pred,squared=False)

print('loss',loss)
print('rmse',rmse)
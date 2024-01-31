#5일분(720행)을 훈련시켜서 하루뒤(144행)을 예측한다
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense,SimpleRNN,Bidirectional,GRU,LSTM,Conv1D,Flatten
from sklearn.model_selection import train_test_split

#1. 데이터
path = "c:/_data/kaggle/jena/"
train = pd.read_csv(path+"jena_climate_2009_2016.csv",index_col=0)
datasets = train
y_col = train['T (degC)']
col = datasets.columns
datasets = pd.DataFrame(datasets,columns=col)
size = 144*3
pred_step = 144


def split_xy(data, time_step, y_col, pred_step):
    result_x = []
    result_y = []
    
    num = len(data) - (time_step + pred_step)
    for i in range(num):
        result_x.append(data[i:i+time_step])
        result_y.append(data.iloc[i+time_step+pred_step][y_col])
        
    return np.array(result_x), np.array(result_y)

# time_step은 5일(720행), pred_step은 1일(144행)로 수정
x, y = split_xy(datasets, size, 'T (degC)', pred_step)


x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.9,random_state=333)

#2.모델구성
model=Sequential()
model.add(Bidirectional(LSTM(units=1),input_shape=(72,14)))
# model.add(Conv1D(filters=1,kernel_size=2,input_shape=(72,14)))
# model.add(Flatten())
model.add(Dense(58))
model.add(Dense(82))
model.add(Dense(85))
model.add(Dense(85))
model.add(Dense(77))
model.add(Dense(69))
model.add(Dense(74))
model.add(Dense(1))

#3.컴파일 훈련
from keras.callbacks import EarlyStopping,ModelCheckpoint
es= EarlyStopping(monitor='loss',mode='auto',patience=5,verbose=3,restore_best_weights=True)
model.compile(loss='mse',optimizer='adam')
model.fit(x_train,y_train,epochs=10,batch_size=1000,callbacks=[es])

#4.결과예측
result = model.evaluate(x_test,y_test)
y_pred=model.predict([x_test])
print(y_pred.shape)
print("결과",y_pred)
print(("loss",result))
#145번 24시간

'''
Bidirectional.GRU   ('loss', 0.27828383445739746)
Bidirectional.LSTM  ('loss', 0.18857364356517792)
GRU                 ('loss', 0.9661486744880676)
LSTM                ('loss', 0.4881895184516907)

(42034, 1)
결과 [[14.969286 ]
 [ 3.0928538]
 [ 3.0924962]
 ...
 [ 3.0924962]
 [ 3.0947528]
 [14.969285 ]]
('loss', 36.35129165649414)
'''



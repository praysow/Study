
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense,SimpleRNN,Bidirectional,GRU
from sklearn.model_selection import train_test_split

#1. 데이터
path = "c:/_data/kaggle/jena/"
train = pd.read_csv(path+"jena_climate_2009_2016.csv",index_col=0)
# TC = pd.read_csv(path+"jena_climate_2009_2016.csv",index_col=0)
# TC = TC.iloc[:,2]
datasets = train
y_col = train['T (degC)']
col = datasets.columns
datasets = pd.DataFrame(datasets,columns=col)
# a= np.array(range(1,11))
size = 144    #bbb의 열의 갯수


def split_xy(data, time_step, y_col):
    result_x = []
    result_y = []
    
    num = len(data) - time_step
    for i in range(num):
        result_x.append(data[i : i+time_step])
        result_y.append(data.iloc[i+time_step][y_col])
    
    return np.array(result_x), np.array(result_y)

x, y = split_xy(datasets,3,'T (degC)')

# print(x.shape,y.shape)  #(6, 4) (6,)

x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.9,random_state=333)

# print(x_train.shape,y_train.shape)  #(378366, 144, 13) (378366, 13)
# print(x_test.shape,y_test.shape)    #(42041, 144, 13) (42041, 13)

#2.모델구성
model=Sequential()
model.add(GRU(units=10,input_shape=(3,14)))
model.add(Dense(15))
model.add(Dense(10))
model.add(Dense(8))
model.add(Dense(1))

# model.summary()
#3.컴파일 훈련
from keras.callbacks import EarlyStopping,ModelCheckpoint
es= EarlyStopping(monitor='loss',mode='auto',patience=50,verbose=3,restore_best_weights=True)
model.compile(loss='mse',optimizer='adam')
model.fit(x_train,y_train,epochs=10,batch_size=1000,callbacks=[es])

#4.결과예측
result = model.evaluate(x_test,y_test)
y_pred=model.predict([x_test])
print("결과",y_pred)
print(("loss",result))
#145번 24시간

'''
Bidirectional.GRU   ('loss', 0.27828383445739746)
Bidirectional.LSTM  ('loss', 0.18857364356517792)
GRU                 ('loss', 0.9661486744880676)
LSTM                ('loss', 0.4881895184516907)
'''



import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, GRU, Bidirectional,LSTM
from sklearn.model_selection import train_test_split

# 1. 데이터
path = "c:/_data/kaggle/jena/"
train = pd.read_csv(path + "jena_climate_2009_2016.csv", index_col=0)
datasets = train
y_col = train['T (degC)']
col = datasets.columns
datasets = pd.DataFrame(datasets, columns=col)
size = 72

def split_xy(data, time_step, y_col, pred_step):
    result_x = []
    result_y = []
    
    num = len(data) - (time_step + pred_step)
    for i in range(num):
        result_x.append(data.iloc[i:i+time_step,:])
        result_y.append(data.iloc[i+time_step:i+time_step+pred_step][y_col].values)
        
    return np.array(result_x), np.array(result_y)

# time_step은 5일(720행), pred_step은 1일(144행)로 수정
x, y = split_xy(datasets, size, 'T (degC)', pred_step=14)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, random_state=333)

# 2. 모델 구성
# model = Sequential()
# model.add(Bidirectional(GRU(units=10), input_shape=(size, 14)))
# # model.add(Dense(15))
# # model.add(Dense(10))
# model.add(Dense(8))
# model.add(Dense(1))

model=Sequential()
model.add(Bidirectional(LSTM(units=1),input_shape=(72,14)))
model.add(Dense(15,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(15,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(1))
# 3. 컴파일 및 훈련
from keras.callbacks import EarlyStopping

es = EarlyStopping(monitor='loss', mode='auto', patience=5, verbose=3, restore_best_weights=True)
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=10, batch_size=1000, callbacks=[es])

# 4. 결과 예측
result = model.evaluate(x_test, y_test)
y_pred = model.predict(x_test)
print("결과", y_pred)
print(("loss", result))

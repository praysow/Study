import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense

a = np.array(range(1, 101))
x_predict = np.array(range(96, 106))
size = 5  # x데이터는 4개, y데이터는 1개

def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i: (i + size)]
        aaa.append(subset)
    return np.array(aaa)

bbb = split_x(a, size)

x = bbb[:, :-1]
y = bbb[:, -1]
x= x.reshape(-1,2,2)
# x_predict = x_predict[-(size-1):].reshape(1, 2, 2)
x_predict = x_predict.reshape(-1, 2, 2)

print(x_predict.shape)

# model = Sequential()
# model.add(LSTM(units=10, input_shape=(2, 2)))
# model.add(Dense(500))
# model.add(Dense(600))
# model.add(Dense(700))
# model.add(Dense(300))
# model.add(Dense(1))

# model.compile(loss='mse', optimizer='adam')
# model.fit(x, y, epochs=1000)


# # 결과 예측
# result = model.evaluate(x, y)
# y_pred = model.predict(x_predict)

# print("loss", result)
# print("예측값", y_pred)

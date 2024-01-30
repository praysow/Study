import numpy as np
from keras.models import Sequential
from keras.layers import LSTM,Dense
from sklearn.model_selection import train_test_split
a= np.array(range(1,101))
x_predict = np.array(range(96,106))
size = 5    # x데이터는 4개, y데이터는 1개

def split_x(dataset,size):
    aaa = []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i : (i+size)]
        aaa.append(subset)      #append는 이어서 붙인다는뜻
    return np.array(aaa)

bbb= split_x(a,size)
# print("bbb",bbb)
# print(bbb.shape)    #(96, 5)

x = bbb[:,:-1]
y = bbb[:,-1]
# print(x,y)

# print(x.shape,y.shape)  #(96, 4) (96,)
# print(x_predict.shape)
# print(x_train.shape,y_train.shape)  #(96, 4) (96,)
# print(x_test.shape,y_test.shape)  #(96, 4) (96,)

# 2.모델구성
model=Sequential()
model.add(LSTM(units=10, input_shape=(4,1)))
model.add(Dense(7))


#3.컴파일 훈련
model.compile(loss='mse',optimizer='adam')
model.fit(x,y,epochs=1)

x_predict = x_predict.reshape(-1, size -1, 1)

#4.결과예측
result= model.evaluate(x,y)
y_pred= model.predict(x_predict)
print("loss",result)
print("예측값",y_pred)








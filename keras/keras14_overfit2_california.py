from sklearn.datasets import fetch_california_housing
datasets = fetch_california_housing()

x = datasets.data
y = datasets.target

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
import numpy as np

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.7,random_state=100)
model=Sequential()
model.add(Dense(1,input_dim=8))
model.add(Dense(100))
model.add(Dense(1))
model.add(Dense(100))
model.add(Dense(1))
model.add(Dense(100))
model.add(Dense(1))
model.add(Dense(100))
model.add(Dense(1))


#3.컴파일 훈련
model.compile(loss='mae',optimizer='adam')
hist = model.fit(x_train,y_train, epochs=300, batch_size=100,validation_split=0.2,verbose=2)

#4.결과예측
loss=model.evaluate(x_test,y_test)
y_predict=model.predict([x_test])
result=model.predict(x)
# print("x 예측값",result)

import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
r2=r2_score(y_test,y_predict)
print("R2 score",r2)
print("로스 :",loss)

def RMSE(x_train,y_train):
    return np.sqrt(mean_squared_error(y_test,y_predict))
rmse = RMSE(y_test,y_predict)
print("RMSE :",rmse)

import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
font_path = "c:\windows\Fonts\gulim.ttc"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)

plt.figure(figsize=(9,6))
# plt.scatter(hist.history['loss'])
plt.plot(hist.history['loss'],c='red', label='loss',marker='.')
plt.plot(hist.history['val_loss'],c='blue', label='loss',marker='.')
# plt.plot(hist.history['r2'],c='pink', label='loss',marker='.')
plt.legend(loc='upper right')
plt.title('켈리포니아 loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.grid()
plt.show()



'''
R2 score -0.04387692066194271
로스 : 0.8775126934051514
RMSE : 1.1761003247890194

'''
from sklearn.datasets import load_diabetes

#1.데이터
datasets= load_diabetes()
x= datasets.data
y= datasets.target

# print(x.shape) #(442,10)
# print(y.shape) #(442,)

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
import numpy as np

x_train,x_test,y_train,y_test= train_test_split(x,y,train_size=0.9, random_state=10)
#2.모델구성
model=Sequential()
model.add(Dense(1,input_dim=10))
model.add(Dense(10))
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(1))
model.add(Dense(10))
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(1))


#3.컴파일 훈련
model.compile(loss='mae',optimizer='adam')
hist=model.fit(x_train,y_train, epochs=1000, batch_size=40, validation_split=0.3, verbose=2)

#4.결과예측
loss=model.evaluate(x_test,y_test)
y_predict=model.predict([x_test])
result=model.predict(x)
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score
r2=r2_score(y_test,y_predict)
print("로스 ,",loss)
print("R2 score :",r2)

from matplotlib import font_manager, rc
font_path = "c:\windows\Fonts\gulim.ttc"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)


plt.figure(figsize=(65,6))
# plt.scatter(hist.history['loss'])
plt.plot(hist.history['loss'],c='red', label='loss',marker='.')
plt.plot(hist.history['val_loss'],c='blue', label='loss',marker='.')
# plt.plot(hist.history['r2'],c='pink', label='loss',marker='.')
plt.legend(loc='upper right')
plt.title('데이콘 로스')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.grid()
plt.show()









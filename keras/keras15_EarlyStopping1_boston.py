#14_1 카피
from sklearn.datasets import load_boston
datasets = load_boston()

x = datasets.data
y = datasets.target

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
import numpy as np

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.7,random_state=100)
model=Sequential()
model.add(Dense(1,input_dim=13))
model.add(Dense(100))
model.add(Dense(1))
model.add(Dense(100))
model.add(Dense(1))
model.add(Dense(100))
model.add(Dense(1))
model.add(Dense(100))
model.add(Dense(1))


#3.컴파일 훈련
from keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss',
                   mode='min',          #min, max, auto
                   patience=50,
                   verbose=1,
                   restore_best_weights=True
                   )
model.compile(loss='mae',optimizer='adam')
hist = model.fit(x_train,y_train, epochs=1000, batch_size=10,validation_split=0.2,verbose=2,
                 callbacks=[es])

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

from matplotlib import font_manager, rc
font_path = "c:\windows\Fonts\gulim.ttc"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)


# plt.figure(figsize=(9,6))
# # plt.scatter(hist.history['loss'])
# plt.plot(hist.history['loss'],c='red', label='loss',marker='.')
# plt.plot(hist.history['val_loss'],c='blue', label='loss',marker='.')
# # plt.plot(hist.history['r2'],c='pink', label='loss',marker='.')
# plt.legend(loc='upper right')
# plt.title('보스턴 로스')
# plt.xlabel('epochs')
# plt.ylabel('loss')
# plt.grid()
# plt.show()


'''
R2 score 0.49506643031068887
로스 : 4.642240047454834
RMSE : 7.151375541012565

R2 score 0.5615699533613386
로스 : 4.391086101531982
RMSE : 6.663810172886282

R2 score 0.539452201116841
로스 : 4.236944675445557
RMSE : 6.829828803429588

R2 score 0.45080646610189934
로스 : 4.513972759246826
RMSE : 7.458219657392365
'''
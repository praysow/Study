from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import pandas as pd

#1. 데이터

path= "c:\_data\dacon\ddarung\\"
train_csv = pd.read_csv(path+"train.csv",index_col=0)
test_csv= pd.read_csv(path+"test.csv",index_col=0)
submission_csv= pd.read_csv(path+"submission.csv")

# print("train",train_csv.shape)      #(1459, 10)
# print("test",test_csv.shape)       #(715, 9)
# print("sub",submission_scv.shape) #(715, 2)

train_csv = train_csv.dropna()
#train_csv=train_csv.fillna(train_csv.mean())                         #test는 dropna를 하면 안되고 결측치를 변경해줘야한다
# train_csv=train_csv.fillna(0)
test_csv=test_csv.fillna(test_csv.mean())                         #test는 dropna를 하면 안되고 결측치를 변경해줘야한다
#test_csv=test_csv.fillna(0)


#######x와 y분리#######
x= train_csv.drop(['count'],axis=1)
y= train_csv['count']



x_train,x_test,y_train,y_test=train_test_split(x,y, train_size=0.89, random_state=50)

#2. 모델구성
# model=Sequential()
# model.add(Dense(1,input_dim=9))
# model.add(Dense(100))
# model.add(Dense(1))
# model.add(Dense(100))
# model.add(Dense(1))
# model.add(Dense(100))
# model.add(Dense(1))
# model.add(Dense(100))
# model.add(Dense(1))




model=Sequential()
model.add(Dense(40,input_dim=9,activation='relu'))
model.add(Dense(80,activation='relu'))
model.add(Dense(120,activation='relu'))
model.add(Dense(160,activation='relu'))
model.add(Dense(200,activation='relu'))
model.add(Dense(240,activation='relu'))
model.add(Dense(300,activation='relu'))
model.add(Dense(1))



#3.컴파일,훈련
from keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss',
                   mode='min',
                   patience=150,
                   verbose=1,
                  restore_best_weights=True
                   )
model.compile(loss='mse',optimizer='adam',metrics=['mae'])
hist = model.fit(x_train, y_train, epochs=5500, batch_size=23,
                 validation_split=0.311,
                 verbose=2,callbacks=[es])

#4.결과예측
loss = model.evaluate(x_test,y_test)
y_submit=model.predict(test_csv)
result=model.predict(x)
#submission.csv 만들기 count컬럼에 값만 넣어주기
submission_csv['count'] = y_submit
print(submission_csv)
submission_csv.to_csv(path + "submission+val_9.csv", index=False)
y_predict=model.predict(x_test)

# import matplotlib.pyplot as plt

r2=r2_score(y_test,y_predict)
print("R2 score",r2)
print("로스 :",loss)
def RMSE(y_test, y_predict):
    # np.sqrt(mean_squared_error(y_test,y_predict))
    return np.sqrt(mean_squared_error(y_test,y_predict))
rmse=RMSE(y_test,y_predict)
print("RMSE :",rmse)

import matplotlib.pyplot as plt
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

'''
R2 score 0.6451265365645753
로스 : [2016.806884765625, 31.689088821411133]
RMSE : 44.90887393795904

'''
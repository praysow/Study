from sklearn.metrics import r2_score, mean_squared_error, mean_squared_log_error
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import pandas as pd

#1. 데이터

path= "c:\_data\kaggle\\bike\\"
train_csv = pd.read_csv(path+"train.csv",index_col=0)
test_csv = pd.read_csv(path+"test.csv",index_col=0)
sampleSubmission_csv = pd.read_csv(path+"sampleSubmission.csv")

# print("train",train_csv.shape)      #(10886, 11)
# print("test",test_csv.shape)       #(6493, 8)
# print("sub",sampleSubmission_csv.shape) #(6493, 2)

x= train_csv.drop(['count','casual','registered'], axis=1)
y= train_csv['count']

x_train,x_test,y_train,y_test=train_test_split(x,y, train_size=0.7, random_state=6)

#2.모델구성
model=Sequential()
model.add(Dense(10,input_dim=8,activation='relu'))
model.add(Dense(9,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(7,activation='relu'))
model.add(Dense(6,activation='relu'))
model.add(Dense(5,activation='relu'))
model.add(Dense(4,activation='relu'))
model.add(Dense(3,activation='relu'))
model.add(Dense(2,activation='relu'))
model.add(Dense(1))


#3.컴파일 훈련
from keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss',
                   mode='min',
                   patience=500,
                   verbose=1,
                   restore_best_weights=True
                   )
model.compile(loss='mse',optimizer='adam')
hist=model.fit(x_train,y_train, epochs=500, batch_size=200,verbose=2,validation_split=0.3,
            callbacks=[es]
               )

#4.결과예측
loss=model.evaluate(x_test,y_test)
y_submit=model.predict(test_csv)
# y_submit=abs(model.predict(test_csv))

sampleSubmission_csv['count'] = y_submit
print(sampleSubmission_csv)
sampleSubmission_csv.to_csv(path +"sampleSubmission_16.csv", index=False)
# print("로스 :",loss)
# print("음수 개수:",sampleSubmission_csv[sampleSubmission_csv['count']<0].count())

y_predict= model.predict(x_test)
r2=r2_score(y_test,y_predict)
print("R2 score",r2)
print("로스 :",loss)
print("음수 개수:",sampleSubmission_csv[sampleSubmission_csv['count']<0].count())
def RMSE(y_test, y_predict):
    #np.sqrt(mean_squared_error(y_test,y_predict))
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
plt.title('케글 로스')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.grid()
plt.show()

'''

R2 score 0.33524081871060485        patience=50
로스 : 21393.99609375               epochs=1000, batch_size=200,verbose=2,validation_split=0.3
음수 개수: datetime    0            15,20,100,150,80,40,20,1
count       0
dtype: int64
RMSE : 146.2668516736104


'''
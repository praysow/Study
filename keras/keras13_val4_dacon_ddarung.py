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

#train_csv = train_csv.dropna()
train_csv=train_csv.fillna(train_csv.mean())                         #test는 dropna를 하면 안되고 결측치를 변경해줘야한다
# train_csv=train_csv.fillna(0)
test_csv=test_csv.fillna(test_csv.mean())                         #test는 dropna를 하면 안되고 결측치를 변경해줘야한다
#test_csv=test_csv.fillna(0)


#######x와 y분리#######
x= train_csv.drop(['count'],axis=1)
y= train_csv['count']



x_train,x_test,y_train,y_test=train_test_split(x,y, train_size=0.8, random_state=50)

#2. 모델구성
model=Sequential()
model.add(Dense(15,input_dim=9,activation='relu'))
model.add(Dense(20,activation='relu'))
model.add(Dense(100,activation='relu'))
model.add(Dense(150,activation='relu'))
model.add(Dense(80,activation='relu'))
model.add(Dense(40,activation='relu'))
model.add(Dense(20,activation='relu'))
model.add(Dense(1))



#3.컴파일,훈련
model.compile(loss='mse',optimizer='adam')
model.fit(x_train, y_train, epochs=1200, batch_size=23,
          #validation_split=0.3,
          verbose=2)

#4.결과예측
loss = model.evaluate(x_test,y_test)
y_submit=model.predict(test_csv)
result=model.predict(x)
#submission.csv 만들기 count컬럼에 값만 넣어주기
submission_csv['count'] = y_submit
print(submission_csv)
submission_csv.to_csv(path + "submission+val_3.csv", index=False)
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
'''
로스 : 2683.97607421875


R2 score 0.6926511999588612         random=50
로스 : 1942.614501953125        epcohs=1000
RMSE : 44.0750993668174         1595:63점  submission+val_1.csv

R2 score 0.7032614972647175     random=50
로스 : 1875.551513671875        epcohs=1200
RMSE : 43.307637173917094

R2 score 0.7203987181113998     random=50
로스 : 1767.2347412109375       epochs=1200
RMSE : 42.03849099013609        "submission+val_2.csv"



'''
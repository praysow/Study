from sklearn.metrics import r2_score, mean_squared_error, mean_squared_log_error
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

#1. 데이터

path= "c:\_data\dacon\cancer\\"
train_csv = pd.read_csv(path+"train.csv",index_col=0)
test_csv = pd.read_csv(path+"test.csv",index_col=0)
sampleSubmission_csv = pd.read_csv(path+"sample_Submission.csv")

print("train",train_csv.shape)      #(652,9)
print("test",test_csv.shape)       #(116, 8)
print("sub",sampleSubmission_csv.shape) #(116,2)

x= train_csv.drop(['Outcome'], axis=1)
y= train_csv['Outcome']
x_train,x_test,y_train,y_test=train_test_split(x,y, train_size=0.9, random_state=8)




#2.모델구성
model=Sequential()
model.add(Dense(10,input_dim=8))
model.add(Dense(40))
model.add(Dense(70))
model.add(Dense(100))
model.add(Dense(130))
model.add(Dense(200))
model.add(Dense(130))
model.add(Dense(100))
model.add(Dense(70))
model.add(Dense(40))
model.add(Dense(10))
model.add(Dense(1,activation='sigmoid'))


# 3.컴파일 훈련
from keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_accuracy',
                   mode='max',
                   patience=10,                   
                   verbose=1,
                   restore_best_weights=True
                   )
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

hist=model.fit(x_train,y_train, epochs=1000, batch_size=10,verbose=2,
               validation_split=0.3, callbacks=[es]
               )

#4.결과예측
loss=model.evaluate(x_test,y_test)
y_submit=model.predict(test_csv)
# y_submit=abs(model.predict(test_csv))

sampleSubmission_csv['Outcome'] = np.round(y_submit)
print(sampleSubmission_csv)
sampleSubmission_csv.to_csv(path +"제출_12.csv", index=False)
# print("로스 :",loss)
# print("음수 개수:",sampleSubmission_csv[sampleSubmission_csv['count']<0].count())

loss,accuracy=model.evaluate(x_test,y_test)
y_predcit=model.predict([x_test])
result=model.predict(x)

def ACC(x_train,y_train):
    return accuracy_score(y_test,np.round(y_predcit))
acc = ACC(y_test,y_predcit)
print("ACC :",acc)
print("로스 :",loss)


# print(y_test)
# print(y_predcit)
# import matplotlib.pyplot as plt
# from matplotlib import font_manager, rc
# font_path = "c:\windows\Fonts\gulim.ttc"
# font = font_manager.FontProperties(fname=font_path).get_name()
# rc('font', family=font)


# plt.figure(figsize=(9,6))
# # plt.scatter(hist.history['loss'])
# plt.plot(hist.history['accuracy'],c='red', label='accuracy',marker='.')
# plt.plot(hist.history['val_accuracy'],c='purple', label='val_accuracy',marker='.')
# # plt.plot(hist.history['r2'],c='pink', label='loss',marker='.')
# plt.legend(loc='upper right')
# plt.title('cancer')
# plt.xlabel('epochs')
# plt.ylabel('loss')
# plt.grid()
# plt.show()

'''

ACC : 0.803030303030303         val_loss        epochs=1000, batch_size=10, validation_split=0.3         patience=50       제출6번
로스 : 0.48295560479164124      10,40,70,100,130,200,130,100,70,40,10       train_size=0.9, random_state=50     42등 0.8103448276점

ACC : 0.8055555555555556        train=0.89      제출7번
로스 : 0.47012457251548767

ACC : 0.8235294117647058        train=0.87    제출8번
로스 : 0.4498840868473053

ACC : 0.8                       train=0.87  validation=0.35
로스 : 0.4697054624557495       9번

'''

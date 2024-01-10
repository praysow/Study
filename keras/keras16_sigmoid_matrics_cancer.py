import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import accuracy_score
#1. 데이터
datasets= load_breast_cancer()
# print(datasets)
# print(datasets.DESCR)
#print(datasets.feature_names)
x = datasets.data       #(569, 30)
y = datasets.target     #(569,)
# print(np.unique(y, return_counts=True))
# (array([0, 1]), array([212, 357], dtype=int64))
# print(x.shape,y.shape)
# print(pd.Series.unique(y))
# print(pd.Series.value_counts(y))
# print(pd.DataFrame(y).value_counts())
# print(pd.value_counts(y))


x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8, random_state=450)

#2.모델구성
model=Sequential()
model.add(Dense(10,input_dim=30))
model.add(Dense(100))
model.add(Dense(110))
model.add(Dense(120))
model.add(Dense(130))
# model.add(Dense(5))
# model.add(Dense(4))
# model.add(Dense(3))
# model.add(Dense(2))
model.add(Dense(1, activation='sigmoid'))


#3.컴파일 훈련
from keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_accuracy',
                   mode='max',
                   patience=50,                   
                   verbose=1,
                   restore_best_weights=True
                   )
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])     #metrics에는 여러개 사용가능, accuracy는 정확도
#'binary_crossentropy' 이진분류일때 사용
hist = model.fit(x_train, y_train, epochs=10, batch_size=10, validation_data=(x_test, y_test),
                 verbose=2, callbacks=[es])

#4.결과예측
loss,accuracy=model.evaluate(x_test,y_test)
y_predcit=model.predict([x_test])
result=model.predict(x)

import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
r2=r2_score(y_test,y_predcit)
print("로스:",loss)
print("R2 score",r2)
print("accuracy :",accuracy)
# print(y_predcit)
# print(y_test)

def ACC(x_train,y_train):
    return accuracy_score(y_test,np.around(y_predcit,1))
acc = ACC(y_test,y_predcit)
print("ACC :",acc)




# from matplotlib import font_manager, rc
# font_path = "c:\windows\Fonts\gulim.ttc"
# font = font_manager.FontProperties(fname=font_path).get_name()
# rc('font', family=font)


# plt.figure(figsize=(9,6))
# # plt.scatter(hist.history['loss'])
# plt.plot(hist.history['loss'],c='red', label='loss',marker='.')
# plt.plot(hist.history['val_loss'],c='blue', label='val_loss',marker='.')
# plt.plot(hist.history['val_accuracy'],c='purple', label='val_accuracy',marker='.')
# # plt.plot(hist.history['r2'],c='pink', label='loss',marker='.')
# plt.legend(loc='upper right')
# plt.title('유방암 로스')
# plt.xlabel('epochs')
# plt.ylabel('loss')
# plt.grid()
# plt.show()

'''
accuracy: 0.9649
로스: [0.1454794704914093, 0.9649122953414917]
R2 score 0.8338768633346818
RMSE : 0.2037596661029616

'''


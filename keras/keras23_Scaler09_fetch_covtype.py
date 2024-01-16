import numpy as np
import pandas as pd
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder

#1.데이터
datasets= fetch_covtype()
x= datasets.data
y= datasets.target
y_ohe1= to_categorical(datasets.target)
y_ohe1= y_ohe1[:,1:]
# print(y_ohe1.shape) (581012, 8)
# print(np.unique(y,return_counts=True))



# for i in range(8):
#     print(np.unique(y_ohe1[:,i],return_counts=True))               

# print(y_ohe1.shape)

# print(x.shape,y.shape)      # (581012, 54) (581012,)
# print(pd.value_counts(y))
#1    71
#0    59
#2    48
# #pandas
y_ohe2 = pd.get_dummies(y)
# print(y_ohe2.shape)       (581012, 7)
# # print(y_ohe2.shape)


# # 사이킷런
y=y.reshape(-1,1)

ohe = OneHotEncoder(sparse=False)
ohe = OneHotEncoder()
y_ohe3= ohe.fit_transform(y).toarray()
#print(y_ohe3.shape)     (581012, 7)

print(y_ohe3)

r = int(np.random.uniform(1, 1000))
x_train, x_test, y_train, y_test = train_test_split(x, y_ohe1, train_size=0.86,
                                                    random_state=5,        #346
                                                    # stratify=y_ohe1            
                                                    )

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


r1=int(np.random.uniform(1,200))
r2=int(np.random.uniform(1,200))
r3=int(np.random.uniform(1,200))
r4=int(np.random.uniform(1,200))
r5=int(np.random.uniform(1,200))
r6=int(np.random.uniform(1,200))
r0=int(np.random.uniform(1,300))

#2.모델구성
model=Sequential()
model.add(Dense(30,input_dim=54,activation='relu'))
model.add(Dense(70,activation='relu'))
model.add(Dense(80,activation='relu'))
model.add(Dense(90,activation='relu'))
model.add(Dense(100,activation='relu'))
model.add(Dense(7, activation='softmax'))

# 3.컴파일 훈련
from keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_accuracy',
                   mode='max',
                   patience=100,
                   verbose=1,
                   restore_best_weights=True
                   )
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics='accuracy')
model.fit(x_train, y_train, epochs=3000,batch_size=60000, validation_split=0.3,verbose=2,
          callbacks=[es]
            )


#4.결과예측
result = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)

y_test = np.argmax(y_test,axis=1)
y_predict= np.argmax(y_predict,axis=1)

# print(y_test)
# print(y_predict)
# print(y_test.shape,y_predict.shape)


from sklearn.metrics import accuracy_score
acc = accuracy_score(y_predict,y_test)




print("accuracy_score :", acc)
print("로스 :", result[0])
print("acc :",result[1])
print("random값 :", r)

print("r1",r1)
print("r2",r2)
print("r3",r3)
print("r4",r4)
print("r5",r5)
print("r0",r0)
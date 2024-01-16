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

x_train, x_test, y_train, y_test = train_test_split(x, y_ohe1, train_size=0.86,
                                                    random_state=5,        #346
                                                    # stratify=y_ohe1            
                                                    )

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


#2.모델구성
model=Sequential()
model.add(Dense(30,input_dim=54,activation='relu'))
model.add(Dense(70,activation='relu'))
model.add(Dense(80,activation='relu'))
model.add(Dense(90,activation='relu'))
model.add(Dense(100,activation='relu'))
model.add(Dense(7, activation='softmax'))

# 3.컴파일 훈련
from keras.callbacks import EarlyStopping, ModelCheckpoint
mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    verbose=1,
    save_best_only=True,
    filepath='..\_data\_save\MCP\keras26_MCP_fetch_covtype.hdf5'
    )
es= EarlyStopping(monitor='val_loss',mode='min',patience=100,verbose=1,restore_best_weights=True)
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics='accuracy')
hist= model.fit(x_train, y_train, epochs=1000,batch_size=100000, validation_split=0.1,verbose=2,
          callbacks=[es,mcp])

model.save("c:\_data\_save\\fetch_covtype_1.h5")

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


'''
accuracy_score : 0.841262816257284
로스 : 0.3754711449146271
acc : 0.8412628173828125
random값 : 892

'''
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd

#1. 데이터
path= "c:\_data\dacon\wine\\"
train_csv = pd.read_csv(path+"train.csv",index_col=0)
test_csv = pd.read_csv(path+"test.csv",index_col=0)
sampleSubmission_csv = pd.read_csv(path+"sample_Submission.csv")
x= train_csv.drop(['quality'], axis=1)
y= train_csv['quality']
# x['type'] = x['type'].map({'red': 1, 'white': 2})
# y['type'] = y['type'].map({'red': 1, 'white': 2})

# print(train_csv.shape)  (5497, 13)
# print(test_csv.shape)   (1000, 12)

y_ohe= pd.get_dummies(y)

# print(y_ohe)
# print(y_ohe.shape)      (5497, 7)

# print(np.unique(y,return_counts=True))
# (array([3, 4, 5, 6, 7, 8, 9], dtype=int64), array([  26,  186, 1788, 2416,  924,  152,    5], dtype=int64))
# x_train,x_test,y_train,y_test= train_test_split(x,y,)
# print(train_csv.isna().sum()) 
# print(test_csv.isna().sum())
lb=LabelEncoder()
lb.fit(x['type'])
x['type'] =lb.transform(x['type'])
test_csv['type'] =lb.transform(test_csv['type'])

while True:
    r=int(np.random.uniform(1000,2000))
    train=np.random.uniform(0.9,0.99)
    x_train,x_test,y_train,y_test=train_test_split(x,y_ohe, train_size=train, random_state=r,
                                                    stratify=y_ohe)

    x_train=np.asarray(x_train).astype(np.float32)
    x_test=np.asarray(x_test).astype(np.float32)
    test_csv = np.asarray(test_csv).astype(np.float32)

    r1=int(np.random.uniform(1,100))
    r2=int(np.random.uniform(50,150))
    r3=int(np.random.uniform(80,180))
    r4=int(np.random.uniform(80,180))
    r5=int(np.random.uniform(90,190))
    r6=int(np.random.uniform(1,200))
    r7=int(np.random.uniform(1,200))
    r8=int(np.random.uniform(1,200))
    r9=int(np.random.uniform(150,200))
    r10=int(np.random.uniform(150,200))

    r0=int(np.random.uniform(1,1000))


    #2.모델구성
    model=Sequential()
    # model.add(Dense(34,input_dim=12,activation='relu'))
    # model.add(Dense(116,activation='relu'))
    # model.add(Dense(112,activation='relu'))
    # model.add(Dense(83,activation='relu'))
    # model.add(Dense(157,activation='relu'))
    # model.add(Dense(188,activation='relu'))
    # model.add(Dense(34,activation='relu'))
    # model.add(Dense(3,activation='relu'))
    # model.add(Dense(174,activation='relu'))
    # model.add(Dense(157,activation='relu'))
    # model.add(Dense(7,activation='softmax'))

    model.add(Dense(r1,input_dim=12,activation='relu'))
    model.add(Dense(r2))
    model.add(Dense(r3,activation='relu'))
    model.add(Dense(r4))
    model.add(Dense(r5,activation='relu'))
    model.add(Dense(r6))
    model.add(Dense(r7,activation='relu'))
    model.add(Dense(r8))
    model.add(Dense(r9))
    model.add(Dense(r10,activation='relu'))
    model.add(Dense(7,activation='softmax'))

    # ACC : 0.43636363636363634
    # 로스 : [1.2733917236328125, 0.4363636374473572]
    # t 0.845499177783265
    # random 7886
    # r1 34
    # r2 116
    # r3 112
    # r4 83
    # r5 157
    # r6 188
    # r7 34
    # r8 3
    # r9 174
    # r10 157


    #3. 컴파일 훈련
    from keras.callbacks import EarlyStopping
    es = EarlyStopping(monitor='val_loss',
                    mode='min',
                    patience=50,
                    verbose=1,
                    restore_best_weights=True
                        )
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics='accuracy')
    hist=model.fit(x_train,y_train,epochs=1000,validation_split=0.1,batch_size=100,verbose=2,
            callbacks=[es]
            )
    hist = model.fit(x_train, y_train, epochs=r0, batch_size=10, verbose=2, validation_split=0.3,
                callbacks=[es]
                )

    # 학습 이력에서 검증 손실을 가져옴
    accuracy = hist.history['val_loss'][-1]

# 특정 조건을 만족하면 루프 종료
    if accuracy < 0.9:  
        break


#4.결과예측
loss= model.evaluate(x_test,y_test)
y_submit= model.predict(test_csv)
y_test= np.argmax(y_test,axis=1)
y_submit = np.argmax(y_submit,axis=1)+3

sampleSubmission_csv['quality'] = (y_submit)
# print(sampleSubmission_csv)
sampleSubmission_csv.to_csv(path + "submission_7.csv", index=False)

y_predict=model.predict(x_test)
y_predict= np.argmax(y_predict,axis=1)          #+3을 한 이유는 y에 유니크를 찍었을때는 3,4,5,6,7,8,9로 나왔는데 
                                                    #y_predict는 0,1,2,3,4,5,6,7 이런식으로 나왔기 때문에 데이콘에 제출을 위해서 뒤로 3칸씩 미는 작업을 한것이다




def ACC(x_train,y_train):
    return accuracy_score(y_test,np.round(y_predict))
acc = ACC(y_test,y_predict)
print("ACC :",acc)
print("로스 :",loss)
print("t",train)
print("random",r)
print("r1",r1)
print("r2",r2)
print("r3",r3)
print("r4",r4)
print("r5",r5)
print("r6",r6)
print("r7",r7)
print("r8",r8)
print("r9",r9)
print("r10",r10)

import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
font_path = "c:\windows\Fonts\gulim.ttc"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)

plt.figure(figsize=(9,6))
# plt.scatter(hist.history['loss'])
plt.plot(hist.history['val_loss'],c='red', label='loss',marker='.')
plt.plot(hist.history['val_accuracy'],c='blue', label='acc',marker='.')
# plt.plot(hist.history['r2'],c='pink', label='loss',marker='.')
plt.legend(loc='upper right')
plt.title('wine loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.grid()
plt.show()

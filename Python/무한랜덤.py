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
while True:
    random=int(np.random.uniform(1,10000))
    train=np.random.uniform(0.7,0.99)
    x_train,x_test,y_train,y_test=train_test_split(x,y, train_size=0.9, random_state=random)

    r1=int(np.random.uniform(1,200))
    r2=int(np.random.uniform(1,200))
    r3=int(np.random.uniform(1,200))
    r4=int(np.random.uniform(1,200))
    r5=int(np.random.uniform(1,200))
    r6=int(np.random.uniform(1,200))
    r0=int(np.random.uniform(1,300))


#2.모델구성
    model=Sequential()
    model.add(Dense(r1,input_dim=8))
    model.add(Dense(r2))
    model.add(Dense(r2))
    model.add(Dense(r3))
    model.add(Dense(r4))
    model.add(Dense(r5))
    model.add(Dense(r6))
    model.add(Dense(1,activation='sigmoid'))


# 3.컴파일 훈련

    from keras.callbacks import EarlyStopping
    es = EarlyStopping(monitor='val_accuracy',
            mode='max',
            patience=50,                   
            verbose=1,
            restore_best_weights=True
            )
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

# model.fit()을 실행하여 학습
    hist = model.fit(x_train, y_train, epochs=r0, batch_size=10, verbose=2, validation_split=0.3,
                callbacks=[es]
                )

    # 학습 이력에서 검증 손실을 가져옴
    accuracy = hist.history['val_accuracy'][-1]

    # 특정 조건을 만족하면 루프 종료
    if accuracy > 0.83:  
        break

# hist=model.fit(x_train,y_train, epochs=1000, batch_size=10,verbose=2,
#           validation_split=0.3, callbacks=[es]
#           )

#4.결과예측
# loss=model.evaluate(x_test,y_test)
# y_submit=model.predict(test_csv)
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
print("random",random)
print("r1",r1)
print("r2",r2)
print("r3",r3)
print("r4",r4)
print("r5",r5)
print("r6",r6)
print("r0",r0)
print("t",train)


#문제점 에폭의 횟수가 계속 같은횟수로 훈련함
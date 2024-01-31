import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers import Dense,Input,Concatenate,concatenate

#1. 데이터
x1_datasets = np.array([range(100),range(301,401)]).T     #삼성전자 종가, 하이닉스 종가
x2_datasets = np.array([range(101,201),range(411,511),range(150,250)]).T  #원유, 환율, 금 시세
x3_datasets = np.array([range(100),range(301,401),range(77,177),range(33,133)]).T
print(x1_datasets.shape,x2_datasets.shape)

y=np.array(range(3001,3101))    #비트코인 종가

x1_train,x1_test,x2_train,x2_test,x3_train,x3_test,y_train,y_test=train_test_split(x1_datasets,x2_datasets,x3_datasets,y,train_size=0.7,random_state=333)

# print(x1_train.shape,x2_train.shape,x3_train.shape,y_train.shape)    #(70, 2) (70, 3) (70, 4) (70,)

#모델 1
input1 = Input(shape= (2,))
dense1= Dense(10,activation='swish',name='bit1')(input1)
dense2= Dense(10,activation='swish',name='bit2')(dense1)
dense3= Dense(10,activation='swish',name='bit3')(dense2)
dense4= Dense(10,activation='swish',name='bit4')(dense3)
output1= Dense(10,activation='swish',name='bit5')(dense4)

# model1 = Model(inputs=input1, outputs= output1)
# model1.summary()

#모델 2
input11 = Input(shape= (3,))
dense11= Dense(100,activation='swish',name='bit11')(input11)
dense12= Dense(100,activation='swish',name='bit12')(dense11)
dense13= Dense(100,activation='swish',name='bit13')(dense12)
dense14= Dense(100,activation='swish',name='bit14')(dense13)
output11= Dense(5,activation='swish',name='bit15')(dense14)

#모델3
input111 = Input(shape= (4,))
dense111= Dense(100,activation='swish',name='bit111')(input111)
dense112= Dense(100,activation='swish',name='bit112')(dense111)
dense113= Dense(100,activation='swish',name='bit113')(dense112)
dense114= Dense(100,activation='swish',name='bit114')(dense113)
output111= Dense(5,activation='swish',name='bit115')(dense114)
# model2 = Model(inputs=input11, outputs= output11) #model.add(Dense(1,output1,output2))
# model2.summary()

#2-3.Concatenate
merge1 = concatenate([output1,output11,output111],name='mg1')
merge2 = Dense(10,name='mg2')(merge1)
merge3 = Dense(11,name='mg3')(merge2)
last_output = Dense(1, name='last')(merge3)

model = Model(inputs=[input1,input11,input111],outputs=last_output)
# model.summary()
#3.컴파일
model.compile(loss= 'mse',optimizer='adam')
model.fit([x1_train,x2_train,x3_train],y_train,epochs=10)

y=y.reshape(100,)
#4.결과예측
result = model.evaluate([x1_test,x2_test,x3_test],y_test)
predict = model.predict([x1_test,x2_test,x3_test])
print("loss:",result)
print("예측값:",predict)

'''
loss: 157193.296875
예측값: [[3335.7917]
 [3445.7705]
 [3787.7373]
 [3470.214 ]
 [3531.3145]
 [2907.3672]
 [3237.9314]
 [3543.5327]
 [3860.718 ]
 [3164.4912]
 [3250.1724]
 [3665.6938]
 [3140.0227]
 [3848.5798]
 [2919.6406]
 [3225.6904]
 [3384.665 ]
 [2870.5874]
 [3763.352 ]
 [3066.6853]
 [3555.7505]
 [3372.4465]
 [2882.8315]
 [3409.1045]
 [3616.8354]
 [3421.326 ]
 [3482.435 ]
 [2774.3267]
 [3201.2083]
 [3103.3428]]

'''
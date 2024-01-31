import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers import Dense,Input,Concatenate,concatenate

#1. 데이터
x1_datasets = np.array([range(100),range(301,401)]).T     #삼성전자 종가, 하이닉스 종가
x2_datasets = np.array([range(101,201),range(411,511),range(150,250)]).T  #원유, 환율, 금 시세

print(x1_datasets.shape,x2_datasets.shape)

y=np.array(range(3001,3101))    #비트코인 종가

# x1_train,x1_test,x2_train,x2_test,y_train,y_test=train_test_split(x1_datasets,x2_datasets,y,train_size=0.7,random_state=333)

# print(x1_train.shape,x2_train.shape,y_train.shape)    #(70, 2) (70, 3) (70,)
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

# model2 = Model(inputs=input11, outputs= output11) #model.add(Dense(1,output1,output2))
# model2.summary()

#2-3.Concatenate
merge1 = concatenate([output1,output11],name='mg1')
merge2 = Dense(10,name='mg2')(merge1)
merge3 = Dense(11,name='mg3')(merge2)
last_output = Dense(1, name='last')(merge3)

model = Model(inputs=[input1,input11],outputs=last_output)
# model.summary()


# __________________________________________________________________________________________________
#  Layer (type)                Output Shape                 Param #   Connected to
# ==================================================================================================
#  input_1 (InputLayer)        [(None, 2)]                  0         []

#  input_2 (InputLayer)        [(None, 3)]                  0         []

#  bit1 (Dense)                (None, 10)                   30        ['input_1[0][0]']

#  bit11 (Dense)               (None, 100)                  400       ['input_2[0][0]']

#  bit2 (Dense)                (None, 10)                   110       ['bit1[0][0]']

#  bit12 (Dense)               (None, 100)                  10100     ['bit11[0][0]']

#  bit3 (Dense)                (None, 10)                   110       ['bit2[0][0]']

#  bit13 (Dense)               (None, 100)                  10100     ['bit12[0][0]']

#  bit4 (Dense)                (None, 10)                   110       ['bit3[0][0]']

#  bit14 (Dense)               (None, 100)                  10100     ['bit13[0][0]']

#  bit5 (Dense)                (None, 10)                   110       ['bit4[0][0]']

#  bit15 (Dense)               (None, 5)                    505       ['bit14[0][0]']

#  mg1 (Concatenate)           (None, 15)                   0         ['bit5[0][0]',            연산량이 0인 이유는 단순히 이어붙였기때문에 연산이 안된것이다
#                                                                      'bit15[0][0]']

#  mg2 (Dense)                 (None, 10)                   160       ['mg1[0][0]']             (10+5)*10+10

#  mg3 (Dense)                 (None, 11)                   121       ['mg2[0][0]']

#  last (Dense)                (None, 1)                    12        ['mg3[0][0]']

# ==================================================================================================
# Total params: 31968 (124.88 KB)
# Trainable params: 31968 (124.88 KB)
# Non-trainable params: 0 (0.00 Byte)
# __________________________________________________________________________________________________
#3.컴파일
model.compile(loss= 'mse',optimizer='adam')
model.fit([x1_datasets,x2_datasets],y,epochs=10)

y=y.reshape(100,)
#4.결과예측
result = model.evaluate([x1_datasets,x2_datasets],y)
predict = model.predict([x1_datasets,x2_datasets])
print("loss:",result)
print("예측값:",predict)



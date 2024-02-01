import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers import Dense,Input,Concatenate,concatenate

#1. 데이터
x1_datasets = np.array([range(100),range(301,401)]).T     #삼성전자 종가, 하이닉스 종가
x2_datasets = np.array([range(101,201),range(411,511),range(150,250)]).T  #원유, 환율, 금 시세
x3_datasets = np.array([range(100),range(301,401),range(77,177),range(33,133)]).T
print(x1_datasets.shape,x2_datasets.shape)

y1=np.array(range(3001,3101))    #비트코인 종가
y2=np.array(range(13001,13101))    #이더리움 종가

x1_train,x1_test,x2_train,x2_test,x3_train,x3_test,y1_train,y1_test,y2_train,y2_test=train_test_split(x1_datasets,x2_datasets,x3_datasets,y1,y2,train_size=0.7,random_state=333)

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

#모델 3
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
merge2 = Dense(100,name='mg2')(merge1)
merge3 = Dense(110,name='mg3')(merge2)
merge4 = Dense(110,name='mg4')(merge3)
merge5 = Dense(110,name='mg5')(merge4)
merge6 = Dense(110,name='mg6')(merge5)
last_output1 = Dense(1, name='last1')(merge6)
last_output2 = Dense(1, name='last2')(merge6)

model = Model(inputs=[input1,input11,input111],outputs=[last_output1,last_output2])
# model.summary()
#3.컴파일
model.compile(loss= 'mse',optimizer='adam')
model.fit([x1_train,x2_train,x3_train],[y1_train,y2_train],epochs=1000)

#4.결과예측
result = model.evaluate([x1_test,x2_test,x3_test],[y1_test,y2_test])
predict = model.predict([x1_test,x2_test,x3_test])
print("loss:",result)
print("예측값:",predict)

'''
loss: [4966.1943359375, 234.9996795654297, 4731.19482421875]
예측값: [array([[3044.2087],
       [3048.571 ],
       [3062.1462],
       [3049.5247],
       [3051.908 ],
       [3026.7827],
       [3040.2021],
       [3052.3872],
       [3065.5686],
       [3037.1326],
       [3040.71  ],
       [3057.2607],
       [3036.1016],
       [3064.8628],
       [3027.2026],
       [3039.6934],
       [3046.1648],
       [3025.5813],
       [3061.1775],
       [3033.0095],
       [3052.867 ],
       [3045.6782],
       [3025.9773],
       [3047.1316],
       [3055.2913],
       [3047.6133],
       [3050.002 ],
       [3020.7036],
       [3038.673 ],
       [3034.5542]], dtype=float32), array([[12984.895 ],
       [12992.559 ],
       [13016.659 ],
       [12994.197 ],
       [12998.299 ],
       [12953.205 ],
       [12977.529 ],
       [12999.131 ],
       [13024.03  ],
       [12971.723 ],
       [12978.4795],
       [13007.868 ],
       [12969.753 ],
       [13022.213 ],
       [12953.751 ],
       [12976.574 ],
       [12988.374 ],
       [12951.823 ],
       [13014.946 ],
       [12963.845 ],
       [12999.969 ],
       [12987.517 ],
       [12952.262 ],
       [12990.067 ],
       [13004.28  ],
       [12990.901 ],
       [12995.016 ],
       [12940.714 ],
       [12974.649 ],
       [12966.789 ]], dtype=float32)]
'''

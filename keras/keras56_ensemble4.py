import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers import Dense,Input,Concatenate,concatenate

#1. 데이터
x1_datasets = np.array([range(100),range(301,401)]).T     #삼성전자 종가, 하이닉스 종가


y1=np.array(range(3001,3101))    #비트코인 종가
y2=np.array(range(13001,13101))    #이더리움 종가

x1_train,x1_test,y1_train,y1_test,y2_train,y2_test=train_test_split(x1_datasets,y1,y2,train_size=0.7,random_state=333)

# print(x1_train.shape,x2_train.shape,y_train.shape)    #(70, 2) (70, 3) (70,)
#모델 1
input1 = Input(shape= (2,))
dense1= Dense(30,activation='swish',name='bit1')(input1)
dense2= Dense(50,activation='swish',name='bit2')(dense1)
dense3= Dense(80,activation='swish',name='bit3')(dense2)
dense4= Dense(70,activation='swish',name='bit4')(dense3)
output1= Dense(1,activation='swish',name='bit5')(dense4)
output2= Dense(1,activation='swish',name='bit6')(dense4)

model = Model(inputs=input1, outputs= [output1,output2])
# model1.summary()


# model2 = Model(inputs=input11, outputs= output11) #model.add(Dense(1,output1,output2))
# model2.summary()
# model.summary()
#3.컴파일
model.compile(loss= 'mse',optimizer='adam')
model.fit(x1_train,[y1_train,y2_train],epochs=1000)

#4.결과예측
result = model.evaluate(x1_test,[y1_test,y2_test])
predict = model.predict(x1_test)
print("loss:",result)
print("예측값:",predict)

'''
loss: [1.9407182931900024, 1.789906620979309, 0.15081170201301575]
예측값: [array([[3053.0056],
       [3061.5325],
       [3088.1318],
       [3063.43  ],
       [3068.1777],
       [3020.1243],
       [3045.4575],
       [3069.1277],
       [3093.8616],
       [3039.8184],
       [3046.3984],
       [3078.6262],
       [3037.9438],
       [3092.9036],
       [3021.068 ],
       [3044.516 ],
       [3056.7908],
       [3017.2825],
       [3086.2283],
       [3032.3245],
       [3070.0776],
       [3055.844 ],
       [3018.232 ],
       [3058.6863],
       [3074.8271],
       [3059.6345],
       [3064.3794],
       [3009.589 ],
       [3042.636 ],
       [3035.1326]], dtype=float32), array([[13052.632 ],
       [13061.82  ],
       [13090.659 ],
       [13063.877 ],
       [13069.026 ],
       [13018.225 ],
       [13044.6045],
       [13070.058 ],
       [13097.004 ],
       [13038.691 ],
       [13045.6   ],
       [13080.351 ],
       [13036.74  ],
       [13095.928 ],
       [13019.223 ],
       [13043.613 ],
       [13056.698 ],
       [13015.178 ],
       [13088.585 ],
       [13030.923 ],
       [13071.089 ],
       [13055.678 ],
       [13016.202 ],
       [13058.742 ],
       [13076.238 ],
       [13059.767 ],
       [13064.906 ],
       [13006.541 ],
       [13041.638 ],
       [13033.825 ]], dtype=float32)]
'''

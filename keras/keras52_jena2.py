import numpy as np
import pandas as pd
from keras.models import Sequential, load_model,Model
from keras.layers import Dense,Dropout,BatchNormalization, AveragePooling2D, Flatten, Conv2D, LSTM, Bidirectional,Conv1D,Input,concatenate
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler, Normalizer, RobustScaler
from sklearn.metrics import accuracy_score, f1_score
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler
#1.데이터
path= "c:/_data/si_hum/"
samsung=pd.read_csv(path+"삼성 240205.csv",encoding='euc-kr',index_col=0)       #삼성전자 액면분할지점 1419번
amore=pd.read_csv(path+"아모레 240205.csv",encoding='euc-kr',index_col=0)       #아모레의 shape는 4350
y=samsung['시가']
y1= y[0:1419]
samsung= samsung[0:1419]
amore=amore[0:1419]
# print(samsung.columns)

# 일부 컬럼만 변경
samsung.rename(columns=lambda x: '전일비투' if x == 'Unnamed: 6' else x, inplace=True)
amore.rename(columns=lambda x: '전일비투' if x == 'Unnamed: 6' else x, inplace=True)

# 변경된 컬럼명 출력
# print(samsung.columns)

lb=LabelEncoder()
lb.fit(samsung['전일비'])
samsung['전일비'] = lb.transform(samsung['전일비'])
lb.fit(samsung['거래량'])
samsung['거래량'] = lb.transform(samsung['거래량'])
lb.fit(samsung['금액(백만)'])
samsung['금액(백만)'] = lb.transform(samsung['금액(백만)'])
lb.fit(amore['전일비'])
amore['전일비'] = lb.transform(amore['전일비'])
lb.fit(amore['거래량'])
amore['거래량'] = lb.transform(amore['거래량'])
lb.fit(amore['금액(백만)'])
amore['금액(백만)'] = lb.transform(amore['금액(백만)'])

numeric_columns = ['시가', '고가', '저가', '종가', '전일비','전일비투', '등락률', '신용비', '개인', '기관', '외인(수량)', '외국계', '프로그램']

for col in numeric_columns:
    samsung[col] = samsung[col].replace({',': ''}, regex=True).astype(int)

for col in numeric_columns:
    amore[col] = amore[col].replace({',': ''}, regex=True).astype(int)

y1 = y1.replace({',': ''}, regex=True).astype(int)

# print(samsung['전일비투'])

# y=pd.get_dummies(y)
# print(samsung.shape,amore.shape,y.shape)    #(10296, 17) (4350, 17) (4350,)
# print(samsung.shape,amore.shape,y.shape)    #(4350, 17) (4350, 17) (4350,)
# z=samsung['시가']
# print(np.unique(z,return_counts=True))
# print(z.shape)
# print(np.unique(y,return_counts=True))
samsung=samsung.astype(np.float32)
amore=amore.astype(np.float32)
y1=y1.astype(np.float32)
size = 30
pred_step = 2


def split_xy(data, time_step, y_col, pred_step):
    result_x = []
    result_y = []
    
    num = len(data) - (time_step + pred_step)
    for i in range(num):
        result_x.append(data[i:i+time_step])
        result_y.append(data.iloc[i+time_step+pred_step][y_col])
        
    return np.array(result_x), np.array(result_y)

# time_step은 5일(720행), pred_step은 1일(144행)로 수정
x1, y = split_xy(samsung, size, '시가', pred_step)
x2, y = split_xy(amore, size, '시가', pred_step)

# y = samsung['시가'][0:4350].astype(float)
print(x1.shape,x2.shape,y.shape)

x1_train,x1_test,x2_train,x2_test,y_train,y_test=train_test_split(x1,x2,y,train_size=0.8,random_state=333)
print(x1_train.shape,x2_train.shape,y_train.shape)


'''
#모델1
input1 = Input(shape= (30,16,))
dense1= Dense(100,activation='swish')(input1)
dense2= Dense(200,activation='swish')(dense1)
dense3= Dense(300,activation='swish')(dense2)
dense4= Dense(400,activation='swish')(dense3)
dense5= Dense(500,activation='swish')(dense4)
dense6= Dense(600,activation='swish')(dense5)
dense7= Dense(500,activation='swish')(dense6)
dense8= Dense(400,activation='swish')(dense7)
dense9= Dense(300,activation='swish')(dense8)
dense10= Dense(200,activation='swish')(dense9)
dense11= Dense(100,activation='swish')(dense10)
output1= Dense(10,activation='swish')(dense11)
#모델2
input11 = Input(shape= (30,16,))
dense11= Dense(100,activation='swish')(input11)
dense12= Dense(200,activation='swish')(dense11)
dense13= Dense(300,activation='swish')(dense12)
dense14= Dense(400,activation='swish')(dense13)
dense15= Dense(500,activation='swish')(dense14)
dense16= Dense(600,activation='swish')(dense15)
dense17= Dense(500,activation='swish')(dense16)
dense18= Dense(400,activation='swish')(dense17)
dense19= Dense(300,activation='swish')(dense18)
dense10= Dense(200,activation='swish')(dense19)
dense11= Dense(100,activation='swish')(dense10)
output11= Dense(10,activation='swish')(dense11)

merge1 = concatenate([output1,output11])
merge2 = Dense(100)(merge1)
merge3 = Dense(110)(merge2)
merge4 = Dense(110)(merge3)
merge5 = Dense(110)(merge4)
merge6 = Dense(110)(merge5)
merge7 = Dense(110)(merge6)
merge8 = Dense(110)(merge7)
last_output = Dense(1, name='last')(merge8)

model = Model(inputs=[input1,input11],outputs=last_output)
model.save("c:\_data\si_hum\삼성전자.h5")

# # model.summary()
# initial_learning_rate = 0.0001
# adam_optimizer = Adam(learning_rate=initial_learning_rate)

# # 학습률 감소 함수 정의 (Step Decay)
# def lr_schedule(epoch):
#     """
#     에포크마다 학습률을 감소시키는 함수
#     """
#     drop_rate = 0.5
#     epochs_drop = 10  # 몇 번의 에포크마다 학습률을 감소시킬 것인지 설정
#     new_learning_rate = initial_learning_rate * np.power(drop_rate, np.floor((1 + epoch) / epochs_drop))
#     return new_learning_rate

# 학습률 스케줄러 콜백 생성
# lr_scheduler = LearningRateScheduler(lr_schedule)
# #3.컴파일
from keras.callbacks import EarlyStopping,ModelCheckpoint
es= EarlyStopping(monitor='val_loss',mode='auto',patience=1000,verbose=1,restore_best_weights=True)
mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    verbose=1,
    save_best_only=True,
    filepath='..\_data\_save\MCP\삼성전자.hdf5'
    )

model.compile(loss='mse',optimizer='adam',metrics=['mae'])
hist= model.fit([x1_train,x2_train], y_train, epochs=10000,batch_size=100, validation_split=0.2,verbose=2,
          callbacks=[es,mcp
                     #,lr_scheduler
                     ]
            )
# model=load_model("c:\_data\_save\dechul_8.h5")
model.save("c:\_data\_save\\삼성전자.h5")
#4.결과예측
result = model.evaluate([x1_test,x2_test],y_test)
predict = model.predict([x1_test,x2_test])
predict = np.round(predict,2)
print("MSE:",result[0])
print("MAE:",result[1])
print("예측값:",predict[0,0],predict[1,0])
# print("예측값:",predict)cc

'''
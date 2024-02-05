#바꾼것 데이터슬라이싱 타임스텝
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
amore=pd.read_csv(path+"아모레 240205.csv",encoding='euc-kr',index_col=0)       #아모레 데이터수 4350
y1=samsung['시가']
y2=amore['종가']
y1=y1[1:1419]
y2=y2[1:1419]
samsung= samsung[1:1419]
amore= amore[1:1419]

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
y2 = y2.replace({',': ''}, regex=True).astype(int)

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
y2=y2.astype(np.float32)

size = 30
pred_step = 1


def split_xy(data, time_step, y_col, pred_step):
    result_x = []
    result_y = []
    
    num = len(data) - (time_step + pred_step)
    for i in range(num):
        result_x.append(data[i:i+time_step])
        result_y.append(data.iloc[i+time_step+pred_step][y_col])
        
    return np.array(result_x), np.array(result_y)

x1, y1 = split_xy(samsung, size, '시가', pred_step)
# x2, y1 = split_xy(amore, size, '시가', pred_step)
# x1, y2 = split_xy(samsung, size, '시가', pred_step)
x2, y2 = split_xy(amore, size, '시가', pred_step)

# y = samsung['시가'][0:4350].astype(float)
print(x1.shape,x2.shape,y1.shape,y2.shape)  #(1387, 30, 16) (1387, 30, 16) (1387,) (1387,)

x1_train,x1_test,x2_train,x2_test,y1_train,y1_test,y2_train,y2_test=train_test_split(x1,x2,y1,y2,train_size=0.8,random_state=3)
# print(x1_train.shape,x2_train.shape,y_train.shape)
x1_train=x1_train.reshape(-1,30*16)
x1_test=x1_test.reshape(-1,30*16)
x2_train=x2_train.reshape(-1,30*16)
x2_test=x2_test.reshape(-1,30*16)
scaler = MinMaxScaler()
scaler.fit(x1_train,x2_train)
x1_train = scaler.transform(x1_train)
x1_test = scaler.transform(x1_test)
x2_train = scaler.transform(x2_train)
x2_test = scaler.transform(x2_test)

x1_train=x1_train.reshape(-1,30,16)
x1_test=x1_test.reshape(-1,30,16)
x2_train=x2_train.reshape(-1,30,16)
x2_test=x2_test.reshape(-1,30,16)

# 모델1
input1 = Input(shape= (size,16,))
dense1= Conv1D(100,2,activation='swish')(input1)
dense2= Conv1D(200,2,activation='swish')(dense1)
dense3= Conv1D(300,2,activation='swish')(dense2)
dense4= Conv1D(400,2,activation='swish')(dense3)
flatten= Flatten()(dense4)
output1= Dense(10,activation='swish')(flatten)

# 모델2
input11 = Input(shape= (size,16,))
dense2_1= Conv1D(100,2,activation='swish')(input11)
dense2_2= Conv1D(200,2,activation='swish')(dense2_1)
dense2_3= Conv1D(200,2,activation='swish')(dense2_2)
dense2_4= Conv1D(300,2,activation='swish')(dense2_3)
flatten= Flatten()(dense2_4)
output11= Dense(10,activation='swish')(flatten)

merge1 = concatenate([output1,output11])
merge2 = Dense(100,activation='swish')(merge1)
merge3 = Dense(110,activation='swish')(merge2)
merge4 = Dense(110,activation='swish')(merge3)
merge5 = Dense(110,activation='swish')(merge4)
merge6 = Dense(110,activation='swish')(merge5)
merge7 = Dense(110,activation='swish')(merge6)
merge8 = Dense(110,activation='swish')(merge7)
merge9 = Dense(100,activation='swish')(merge8)
last_output1 = Dense(1, name='last')(merge9)
last_output2 = Dense(1, name='last2')(merge6)

# 모델 정의 후에 compile 호출이 필요합니다.
model = Model(inputs=[input1,input11],outputs=[last_output1,last_output2])



from keras.callbacks import EarlyStopping,ModelCheckpoint,Callback

class ThresholdCallback(Callback):
    def __init__(self, threshold_percentage=30.0):
        super(ThresholdCallback, self).__init__()
        self.threshold_percentage = threshold_percentage

    def on_epoch_end(self, epoch, logs=None):
        # 훈련 중에 각 에포크가 끝날 때 호출되는 메서드
        # 여기에서 원하는 작업을 수행할 수 있습니다.

        # 예측값을 현재 설정된 threshold_percentage에 따라 조절
        self.model.predictions = self.model.predict([x1_test, x2_test])
        self.model.predictions[0] = np.squeeze(self.model.predictions[0])
        self.model.predictions[1] = np.squeeze(self.model.predictions[1])
        self.model.predictions = np.round(self.model.predictions, 2)
        
        # 예측값을 적용한 결과로 평가
        new_result = self.model.evaluate([x1_test, x2_test], [y1_test, y2_test], verbose=0)

        print("\nEpoch {}: 새로운 평가 결과 - Loss: {:.4f}, MAE: {:.4f}".format(epoch + 1, new_result[0], new_result[1]))


# Callback 인스턴스 생성
threshold_callback = ThresholdCallback(threshold_percentage=30.0)
es= EarlyStopping(monitor='val_loss',mode='auto',patience=70,verbose=1,restore_best_weights=True)
mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    verbose=1,
    save_best_only=True,
    filepath='..\_data\_save\MCP\삼성전자1.hdf5'
    )
# 모델에 콜백 등록
model.compile(loss='mae',optimizer='adam',metrics=['acc'])
model.fit([x1_train, x2_train], [y1_train, y2_train], epochs=10000, batch_size=500, validation_split=0.2, verbose=2,
          callbacks=[es, mcp, threshold_callback])

model.save("c:\_data\_save\\삼성전자1.h5")
#4.결과예측
result = model.evaluate([x1_test,x2_test],[y1_test,y2_test])
predict = model.predict([x1_test,x2_test])

predict = np.round(predict,2)
print("mse:",result)
# print("mse:",result[0])
# print("mae:",result[3])
print("삼성:",predict[0][0])
print("아모레:",predict[1][0])
# print("예측값:",predict)

'''
 loss: 246183472.0000 - last_loss: 87466088.0000 - last2_loss: 158717408.0000 - last_mae: 7788.6382 - last2_mae: 9529.7402 - val_loss: 214116208.0000
 val_last_loss: 74457584.0000 - val_last2_loss: 139658624.0000 - val_last_mae: 7214.2417 - val_last2_mae: 9176.1260 - 138ms/epoch - 69ms/step
loss: [43627832.0, 4841.64599609375]
삼성: [301645.78]
아모레: [232865.56]
'''
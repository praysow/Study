

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
import random as rn
import tensorflow as tf
rn.seed(333)
tf.random.set_seed(123)
np.random.seed(321)
#1.데이터
path= "c:/_data/si_hum/"
samsung=pd.read_csv(path+"삼성 240205.csv",encoding='euc-kr',index_col=0)       #삼성전자 액면분할지점 1419번
amore=pd.read_csv(path+"아모레 240205.csv",encoding='euc-kr',index_col=0)       #아모레 데이터수 4350
y1=samsung['시가']
y2=amore['종가']
y1=y1[0:1419]
y2=y2[0:1419]
samsung= samsung[0:1419]
amore= amore[0:1419]
samsung=samsung.sort_values(['일자'],ascending=[True])
amore=amore.sort_values(['일자'],ascending=[True])
y1=y1.sort_values(ascending=[True])
y2=y2.sort_values(ascending=[True])
z=samsung['시가']
# print(np.unique(z,return_counts=True))
# print(z)

# 일부 컬럼만 변경
samsung.rename(columns=lambda x: '전일비투' if x == 'Unnamed: 6' else x, inplace=True)
amore.rename(columns=lambda x: '전일비투' if x == 'Unnamed: 6' else x, inplace=True)

samsung = samsung.drop(['전일비','전일비투','등락률','금액(백만)','신용비','외인비'],axis=1)
amore = amore.drop(['전일비','전일비투','등락률','금액(백만)','신용비','외인비'],axis=1)

# 변경된 컬럼명 출력
# print(samsung.columns)
# print(samsung.shape,amore.shape)

lb=LabelEncoder()
lb.fit(samsung['거래량'])
samsung['거래량'] = lb.transform(samsung['거래량'])
lb.fit(amore['거래량'])
amore['거래량'] = lb.transform(amore['거래량'])

numeric_columns = ['시가', '고가', '저가', '종가','개인', '기관', '외인(수량)', '외국계', '프로그램']


for col in numeric_columns:
    samsung[col] = samsung[col].replace({',': ''}, regex=True).astype(int)

for col in numeric_columns:
    amore[col] = amore[col].replace({',': ''}, regex=True).astype(int)

y1 = y1.replace({',': ''}, regex=True).astype(int)
y2 = y2.replace({',': ''}, regex=True).astype(int)

samsung=samsung.astype(np.float32)
amore=amore.astype(np.float32)
y1=y1.astype(np.float32)
y2=y2.astype(np.float32)

size = 10
pred_step = 2


def split_xy(data, time_step, y_col, pred_step):
    result_x = []
    result_y = []
    
    num = len(data) - (time_step + pred_step)
    for i in range(num):
        result_x.append(data[i:i+time_step])
        result_y.append(data.iloc[i+time_step+pred_step][y_col])
        
    return np.array(result_x), np.array(result_y)

x1, y1 = split_xy(samsung, size, '시가', pred_step)
x2, y2 = split_xy(amore, size, '종가', pred_step)


x1_train,x1_test,x2_train,x2_test,y1_train,y1_test,y2_train,y2_test=train_test_split(x1,x2,y1,y2,train_size=0.9,random_state=333
                                                                                     ,shuffle=False
                                                                                     )
shape=10

x1_train=x1_train.reshape(-1,size*shape)
x1_test=x1_test.reshape(-1,size*shape)
x2_train=x2_train.reshape(-1,size*shape)
x2_test=x2_test.reshape(-1,size*shape)

scaler = MinMaxScaler()
scaler2= RobustScaler()
scaler.fit(x1_train,x2_train)
scaler2.fit(x1_train,x2_train)
x1_train = scaler2.transform(x1_train)
x1_test = scaler2.transform(x1_test)
x2_train = scaler.transform(x2_train)
x2_test = scaler.transform(x2_test)

x1_train=x1_train.reshape(-1,size,shape)
x1_test=x1_test.reshape(-1,size,shape)
x2_train=x2_train.reshape(-1,size,shape)
x2_test=x2_test.reshape(-1,size,shape)


# 모델1
input1 = Input(shape= (size,shape,))
dense1= Conv1D(100,2,activation='swish',padding='same')(input1)
drop1=Dropout(0.5)(dense1)
dense2= Conv1D(150,2,activation='swish')(drop1)
drop2=Dropout(0.5)(dense2)
dense3= Conv1D(200,2,activation='swish')(drop2)
drop3=Dropout(0.5)(dense3)
dense4= Conv1D(100,2,activation='swish')(drop3)
flatten= Flatten()(dense4)
output1= Dense(10,activation='swish')(flatten)

# 모델2
input11 = Input(shape= (size,shape,))
dense2_1= Conv1D(50,2,activation='swish',padding='same')(input11)
dense2_4= Conv1D(100,2,activation='swish')(dense2_1)
flatten= Flatten()(dense2_4)
output11= Dense(10,activation='swish')(flatten)

merge1 = concatenate([output1,output11])
merge2 = Dense(100,activation='swish')(merge1)
merge3 = Dense(200,activation='swish')(merge2)
merge4 = Dense(250,activation='swish')(merge3)
merge5 = Dense(300,activation='swish')(merge4)
merge6 = Dense(250,activation='swish')(merge5)
merge7 = Dense(200,activation='swish')(merge6)
merge8 = Dense(150,activation='swish')(merge7)
merge9 = Dense(100,activation='swish')(merge8)
last_output1 = Dense(1, name='last')(merge9)
last_output2 = Dense(1, name='last2')(merge3)
 
# 모델 정의 후에 compile 호출이 필요합니다.
model = Model(inputs=[input1,input11],outputs=[last_output1,last_output2])

from keras.callbacks import EarlyStopping,ModelCheckpoint,Callback
방지코드=30.0
class ThresholdCallback(Callback):
    def __init__(self, threshold_percentage=방지코드):        #수치조정
        super(ThresholdCallback, self).__init__()
        self.threshold_percentage = threshold_percentage

    def on_epoch_end(self, epoch, logs=None):
        # 훈련 중에 각 에포크가 끝날 때 호출되는 메서드
        # 여기에서 원하는 작업을 수행할 수 있습니다.

        # 예측값을 현재 설정된 threshold_percentage에 따라 조절
        self.model.predictions = self.model.predict([x1_test, x2_test])
        self.model.predictions[0] = np.squeeze(self.model.predictions[0])
        self.model.predictions[1] = np.squeeze(self.model.predictions[1])

        # 30% 이상 상승 및 30% 이하 감소를 방지하는 코드
        for i in range(2):
            self.model.predictions[i] = np.where(
                self.model.predictions[i] > y1_test * (1 + self.threshold_percentage / 100),
                y1_test * (1 + self.threshold_percentage / 100),
                np.where(
                    self.model.predictions[i] < y1_test * (1 - self.threshold_percentage / 100),
                    y1_test * (1 - self.threshold_percentage / 100),
                    self.model.predictions[i]
                )
            )

        self.model.predictions = np.round(self.model.predictions, 2)
        
        # 예측값을 적용한 결과로 평가
        new_result = self.model.evaluate([x1_test, x2_test], [y1_test, y2_test], verbose=0)

        print("\nEpoch {}: 새로운 평가 결과 - Loss: {:.4f}, MAE: {:.4f}".format(epoch + 1, new_result[0], new_result[1]))


# Callback 인스턴스 생성
threshold_callback = ThresholdCallback(threshold_percentage=방지코드)    #수치조정
es= EarlyStopping(monitor='val_loss',mode='auto',patience=500,verbose=1,restore_best_weights=True)
mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    verbose=1,
    save_best_only=True,
    filepath='c:/_data/si_hum/삼성전자8.hdf5'
    )
# 모델에 콜백 등록
model.compile(loss='mae',optimizer='adam')
model.fit([x1_train, x2_train], [y1_train, y2_train], epochs=100000, batch_size=50, validation_split=0.1, verbose=3,
          callbacks=[es, mcp, threshold_callback])

model.save("c:/_data/si_hum/삼성전자8.h5")
#4.결과예측
result = model.evaluate([x1_test,x2_test],[y1_test,y2_test])
predict = model.predict([x1_test,x2_test])

predict = np.round(predict,2)
print("mae:",result)
print("삼성:",predict[0][-1])
print("아모레:",predict[1][-1])
print("합계:",predict[0][-1]+predict[1][-1])

'''

'''
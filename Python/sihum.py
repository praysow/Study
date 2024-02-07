
import tensorflow as tf
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
path= "c:/_data/sihum/"
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

# 일부 컬럼만 변경
samsung.rename(columns=lambda x: '전일비투' if x == 'Unnamed: 6' else x, inplace=True)
amore.rename(columns=lambda x: '전일비투' if x == 'Unnamed: 6' else x, inplace=True)

samsung = samsung.drop(['전일비','전일비투','등락률','금액(백만)','신용비','외인비'],axis=1)
amore = amore.drop(['전일비','전일비투','등락률','금액(백만)','신용비','외인비'],axis=1)

# print(samsung.columns)
print(samsung.shape,amore.shape)

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

print(x1.shape,x2.shape,y1.shape,y2.shape)

x1_train,x1_test,x2_train,x2_test,y1_train,y1_test,y2_train,y2_test=train_test_split(x1,x2,y1,y2,train_size=0.9,random_state=3
                                                                                     ,shuffle=False
                                                                                     )
print(x1_train.shape,x2_train.shape,y1_train.shape)

shape=10

x1_train=x1_train.reshape(-1,size*shape)
x1_test=x1_test.reshape(-1,size*shape)
x2_train=x2_train.reshape(-1,size*shape)
x2_test=x2_test.reshape(-1,size*shape)
scaler = MinMaxScaler()
scaler2= RobustScaler()
scaler.fit(x1_train)
scaler2.fit(x2_train)
x1_train = scaler.transform(x1_train)
x1_test = scaler.transform(x1_test)
x2_train = scaler2.transform(x2_train)
x2_test = scaler2.transform(x2_test)

x1_train=x1_train.reshape(-1,size,shape)
x1_test=x1_test.reshape(-1,size,shape)
x2_train=x2_train.reshape(-1,size,shape)
x2_test=x2_test.reshape(-1,size,shape)

model=load_model("c:/_data/sihum/삼성전자16.h5")

#4.결과예측
result = model.evaluate([x1_test,x2_test],[y1_test,y2_test])
predict = model.predict([x1_test,x2_test])  

predict = np.round(predict,2)
print("mae:",result)
print("삼성:",predict[0][-1])
print("아모레:",predict[1][-1])
print("합계:",predict[0][-1]+predict[1][-1])

'''
mae: [12909.7255859375, 3755.300537109375, 9154.423828125]
삼성: [74006.52]
아모레: [121132.63]
합계: [195139.16]       16번
'''
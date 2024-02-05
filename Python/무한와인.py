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
amore=pd.read_csv(path+"아모레 240205.csv",encoding='euc-kr',index_col=0)
y=samsung['시가']
y= y[0:4350]
samsung= samsung[0:4350]

lb=LabelEncoder()
# lb.fit(samsung['시가','고가','저가','종가','전일비','등락률','거래3량','금액(백만)','신용비','개인','기관','외인(수량)','외국계','프로그램'])
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

# Preprocess data
def preprocess_data(df):
    lb = LabelEncoder()
    numeric_columns = ['시가', '고가', '저가', '종가', '전일비', '등락률', '신용비', '개인', '기관', '외인(수량)', '외국계', '프로그램']

    for col in numeric_columns:
        df[col] = df[col].replace({',': ''}, regex=True).astype(float)
        lb.fit(df[col])
        df[col] = lb.transform(df[col])

preprocess_data(samsung)
preprocess_data(amore)

y = samsung['시가'][0:4350].astype(float)  # Assuming '시가' is the target variable

# Split data
x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(samsung, amore, y, train_size=0.8, random_state=333)

# Build the model
input1 = Input(shape=(17,))
dense1 = Dense(10, activation='swish', name='bit1')(input1)
dense2 = Dense(10, activation='swish', name='bit2')(dense1)
dense3 = Dense(10, activation='swish', name='bit3')(dense2)
dense4 = Dense(10, activation='swish', name='bit4')(dense3)
output1 = Dense(10, activation='swish', name='bit5')(dense4)

input11 = Input(shape=(17,))
dense11 = Dense(100, activation='swish', name='bit11')(input11)
dense12 = Dense(100, activation='swish', name='bit12')(dense11)
dense13 = Dense(100, activation='swish', name='bit13')(dense12)
dense14 = Dense(100, activation='swish', name='bit14')(dense13)
output11 = Dense(5, activation='swish', name='bit15')(dense14)

merge1 = concatenate([output1, output11], name='mg1')
merge2 = Dense(10, name='mg2')(merge1)
merge3 = Dense(11, name='mg3')(merge2)
last_output = Dense(1, name='last')(merge3)

model = Model(inputs=[input1, input11], outputs=last_output)

# Compile and fit the model
model.compile(loss='mae', optimizer='adam')
model.fit([x1_train, x2_train], y_train, epochs=10)

# Evaluate the model
result = model.evaluate([x1_test, x2_test], y_test)
print("loss:", result)

# Make predictions
predict = model.predict([x1_test, x2_test])
print("예측값:", predict)

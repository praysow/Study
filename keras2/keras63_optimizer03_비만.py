import numpy as np
import pandas as pd
from keras.models import Sequential, load_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler, Normalizer, RobustScaler
from sklearn.metrics import accuracy_score, f1_score
from lightgbm import LGBMClassifier,Booster
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from keras.models import Sequential
from keras.layers import Dense
path= "c:/_data/kaggle/비만/"
train=pd.read_csv(path+"train.csv",index_col=0)
test=pd.read_csv(path+"test.csv",index_col=0)
sample=pd.read_csv(path+"sample_submission.csv")
x= train.drop(['NObeyesdad'],axis=1)
y= train['NObeyesdad']

lb = LabelEncoder()

# 라벨 인코딩할 열 목록
columns_to_encode = ['Gender','family_history_with_overweight','FAVC','CAEC','SMOKE','SCC','CALC','MTRANS']

# 데이터프레임 x의 열에 대해 라벨 인코딩 수행
for column in columns_to_encode:
    lb.fit(x[column])
    x[column] = lb.transform(x[column])

# 데이터프레임 test_csv의 열에 대해 라벨 인코딩 수행
for column in columns_to_encode:
    lb.fit(test[column])
    test[column] = lb.transform(test[column])

y = pd.get_dummies(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, random_state=367, stratify=y,shuffle=True)

scaler =StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
test = scaler.transform(test)

# 모델 

model=Sequential()
model.add(Dense(7,input_shape=(16,),activation='relu'))
model.add(Dense(44,activation='relu'))
model.add(Dense(40,activation='relu'))
model.add(Dense(34,activation='relu'))
model.add(Dense(26,activation='relu'))
model.add(Dense(20,activation='relu'))
model.add(Dense(30,activation='relu'))
model.add(Dense(7,activation='softmax'))

# model= load_model("c:\_data\_save\대출모델9.h5")
#3.컴파일 훈련
from keras.optimizers import Adam
learning_rate = 0.1
model.compile(loss='categorical_crossentropy',optimizer=Adam(learning_rate=learning_rate),metrics=['accuracy'])
hist= model.fit(x_train, y_train, epochs=100,batch_size=3000, validation_split=0.1,verbose=2)

loss = model.evaluate(x_test,y_test)
y_pred = model.predict(x_test)
# 각 출력별로 가장 높은 확률을 가진 클래스 선택
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test, axis=1)
# 정확도 계산
acc = accuracy_score(y_test_classes, y_pred_classes)
print("Accuracy:", acc)
print("lr:{0},acc:{1}".format(learning_rate,acc))

'''
lr:0.1,acc:0.7943159922928709
lr:0.01,acc:0.8819845857418112
lr:0.001,acc:0.8680154142581888
lr:0.0001,acc:0.596820809248555
'''
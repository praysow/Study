import numpy as np
import pandas as pd
from keras.models import Sequential, load_model
from keras.layers import Dense,Dropout,BatchNormalization, AveragePooling1D, Flatten, Conv2D, LSTM, Bidirectional,Conv1D,MaxPooling1D
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler, Normalizer, RobustScaler
from sklearn.metrics import accuracy_score, f1_score
from keras.utils import to_categorical
#1.데이터
path= "c:\_data\dacon\dechul\\"
train_csv=pd.read_csv(path+"train.csv",index_col=0)
test_csv=pd.read_csv(path+"test.csv",index_col=0)
sample_csv=pd.read_csv(path+"sample_submission.csv")
x= train_csv.drop(['대출등급'],axis=1)
y= train_csv['대출등급']


lb=LabelEncoder()
lb.fit(x['대출기간'])
x['대출기간'] = lb.transform(x['대출기간'])
lb.fit(x['근로기간'])
x['근로기간'] = lb.transform(x['근로기간'])
lb.fit(x['주택소유상태'])
x['주택소유상태'] = lb.transform(x['주택소유상태'])
lb.fit(x['대출목적'])
x['대출목적'] = lb.transform(x['대출목적'])

lb.fit(test_csv['대출기간'])
test_csv['대출기간'] =lb.transform(test_csv['대출기간'])

lb.fit(test_csv['근로기간'])
test_csv['근로기간'] =lb.transform(test_csv['근로기간'])

lb.fit(test_csv['주택소유상태'])
test_csv['주택소유상태'] =lb.transform(test_csv['주택소유상태'])

lb.fit(test_csv['대출목적'])
test_csv['대출목적'] =lb.transform(test_csv['대출목적'])


x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.9,random_state=333 ,
                                              #  stratify=y
                                               )

# scaler = RobustScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)
# test_csv = scaler.transform(test_csv)
print(x_train.shape,x_test.shape)
print(y_train.shape,y_test.shape)
from sklearn.svm import LinearSVC,SVC
model = SVC()

model.fit(x_train,y_train)
# # # 4.결과예측
result = model.score(x_test,y_test)
y_predict = model.predict(test_csv)
y_submit = pd.DataFrame(y_predict)
sample_csv["대출등급"]=y_submit
sample_csv.to_csv(path + "대출88.csv", index=False)
y_pred= model.predict(x_test)
f1=f1_score(y_test,y_pred, average='macro')
print("f1",f1)
print("로스:", result)

# '''
# f1 0.9209169259289897
# 로스: 0.18799705803394318
# acc 0.9361370801925659    76번
# '''


#09 boston
#01_캘리포니아
#11_따릉이
#12_케글바이크
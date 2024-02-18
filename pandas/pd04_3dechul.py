import numpy as np
import pandas as pd
from keras.models import Sequential, load_model
from keras.layers import Dense,Dropout,BatchNormalization
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler,RobustScaler
from sklearn.metrics import accuracy_score, f1_score
from keras.utils import to_categorical
from keras.layers import Dense,Flatten,Conv1D
#1.데이터
path= "c:\_data\dacon\dechul\\"
train=pd.read_csv(path+"train.csv",index_col=0)
test=pd.read_csv(path+"test.csv",index_col=0)
sample=pd.read_csv(path+"sample_submission.csv")
x= train.drop(['대출등급', '최근_2년간_연체_횟수', '총연체금액', '연체계좌수'],axis=1)
test= test.drop(['최근_2년간_연체_횟수', '총연체금액', '연체계좌수'],axis=1)
y= train['대출등급']


# print(train_csv,train_csv.shape)        (96294, 14)
# print(test_csv,test_csv.shape)          (64197, 13)
# print(sample_csv,sample_csv.shape)      (64197, 2)
print(np.unique(y,return_counts=True))

lb = LabelEncoder()

# 라벨 인코딩할 열 목록
columns_to_encode = ['대출기간', '근로기간', '주택소유상태','대출목적']

# 데이터프레임 x의 열에 대해 라벨 인코딩 수행
for column in columns_to_encode:
    lb.fit(x[column])
    x[column] = lb.transform(x[column])

for column in columns_to_encode:
    lb.fit(train[column])
    train[column] = lb.transform(train[column])

# 데이터프레임 test_csv의 열에 대해 라벨 인코딩 수행
for column in columns_to_encode:
    lb.fit(test[column])
    test[column] = lb.transform(test[column])

# z= train[ '총상환원금']
# print(pd.value_counts(z))
# # print(np.unique(z,return_counts=True))
# def outliers(data_out):
#     quartile_1,q2,quartile_3 = np.percentile(data_out,[25,50,75])
#     print("1사분위 :", quartile_1)
#     print("q2",q2)
#     print("3사분위 :", quartile_3)
#     iqr = quartile_3 - quartile_1
#     print("iqr:",iqr)
#     lower_bound = quartile_1 - (iqr*1.5)
#     upper_bound = quartile_3 + (iqr*1.5)
#     return np.where((data_out>upper_bound) |
#                     (data_out<lower_bound))

# outliers_loc = outliers(z)
# print("이상치의 위치 :",outliers_loc)

# print(x.columns)Index(['대출금액', '대출기간', '근로기간', '주택소유상태', '연간소득', '부채_대비_소득_비율', '총계좌수', '대출목적',
    #    '최근_2년간_연체_횟수', '총상환원금', '총상환이자', '총연체금액', '연체계좌수']

#대출의 자료특성상 데이터들 마다 수치의 편차가 클수밖에 없어서 이상치라고 보기는 어려웠고
# '최근_2년간_연체_횟수', '총연체금액', '연체계좌수' 이 3가지의 컬럼들이 데이터 전체에 영향은 거의 없을 것 같아서 drop으로 전처리를 함

x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.9,random_state=3,
                                               stratify=y , shuffle=True
                                               )

scaler =RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
test = scaler.transform(test)
import random
r = random.randint(1, 100)
random_state = r
# 모델 생성 및 학습
lgbm_params = {"objective": "multiclass",
               "metric": "multi_logloss",
               "verbosity": -1,
               "boosting_type": "gbdt",
               "random_state": 78,
               "num_class": 7,
               "learning_rate": 0.01386432121252535,
               "n_estimators": 500,
               "feature_pre_filter": False,
               "lambda_l1": 1.2149501037669967e-07,
               "lambda_l2": 0.9230890143196759,
               "num_leaves": 31,
               "feature_fraction": 0.5,
               "bagging_fraction": 0.5523862448863431,
               "bagging_freq": 4,
               "min_child_samples": 20,
               "max_depth":12,
               "min_samples_leaf":10,
               'n_jobs': -1
               }

import lightgbm as lgb

model = lgb.LGBMClassifier(**lgbm_params,device='gpu')
model.fit(x_train, y_train)

# 모델 저장
# booster = model.booster_
# model.booster_.save_model("c:/_data/_save/대출1.h5")

# 테스트 데이터 예측 및 저장
y_pred = model.predict(x_test)
y_submit = model.predict(test)
sample['NObeyesdad'] = y_submit
sample.to_csv(path + "대출1.csv", index=False)

# 정확도 평가
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("r",r)

'''
f1 0.9143798222511382
로스: 0.2211541086435318
acc 0.9261682033538818

Accuracy: 0.8136033229491173
'''
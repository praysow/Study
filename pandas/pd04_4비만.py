import numpy as np
import pandas as pd
from keras.models import Sequential, load_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler, Normalizer, RobustScaler
from sklearn.metrics import accuracy_score, f1_score
from lightgbm import LGBMClassifier,Booster
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor

path= "c:/_data/kaggle/비만/"
train=pd.read_csv(path+"train.csv",index_col=0)
test=pd.read_csv(path+"test.csv",index_col=0)
sample=pd.read_csv(path+"sample_submission.csv")
x= train.drop(['NObeyesdad'],axis=1)
y= train['NObeyesdad']

# print(train.shape,test.shape)   #(20758, 17) (13840, 16)    NObeyesdad
# print(x.shape,y.shape)  #(20758, 16) (20758,)

lb = LabelEncoder()

# 라벨 인코딩할 열 목록
columns_to_encode = ['Gender','family_history_with_overweight','FAVC','CAEC','SMOKE','SCC','CALC','MTRANS']

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

# z= train[
#          'MTRANS'
#         #  'NObeyesdad'
#          ]

# train['Gender'] = train['Gender'].replace({'Female': 0, 'Male': 1})
# print(pd.value_counts(z))
# # print(np.unique(z,return_counts=True))
# train['FAVC'] = train['FAVC'].replace({'yes': 0, 'no': 1})

# def outliers(data_out):
#     quartile_1, q2, quartile_3 = np.percentile(data_out, [25, 50, 75])
#     print("1사분위 :", quartile_1)
#     print("q2", q2)
#     print("3사분위 :", quartile_3)
#     iqr = quartile_3 - quartile_1
#     print("iqr:", iqr)
#     lower_bound = quartile_1 - (iqr * 1.5)    
#     upper_bound = quartile_3 + (iqr * 1.5)
#     return np.where((data_out > upper_bound) | (data_out < lower_bound))

# outliers_loc = outliers(z)
# print("이상치의 위치 :", outliers_loc)
# print(train.columns)   # Index(['Gender', 'Age0', 'Height0', 'Weight', 'family_history_with_overweight0','FAVC0', 'FCVC', 'NCP0', 'CAEC0', 'SMOKE', 'CH2O', 'SCC0', 'FAF', 'TUE','CALC0', 'MTRANS', 'NObeyesdad'

#라벨인코더로 문자를 숫자로 변환하다보니 생긴 이상치로 보임 그래서 더이상의 전처리는 하지않음

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, random_state=367, stratify=y,shuffle=True)

scaler =StandardScaler()
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
               "random_state": random_state,
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


model = lgb.LGBMClassifier(**lgbm_params,device='gpu')
model.fit(x_train, y_train)

# 모델 저장
# booster = model.booster_
model.booster_.save_model("c:/_data/_save/비만31.h5")

# 테스트 데이터 예측 및 저장
y_pred = model.predict(x_test)
y_submit = model.predict(test)
sample['NObeyesdad'] = y_submit
sample.to_csv(path + "비만31.csv", index=False)

# 정확도 평가
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("r",r)

'''
Accuracy: 0.9219653179190751
r 89
'''

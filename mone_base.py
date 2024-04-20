import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import *
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestClassifier
# 데이터 불러오기
path = "c:/_data/dacon/soduc/"
train = pd.read_csv(path+'train.csv', index_col=0)
test = pd.read_csv(path+'test.csv', index_col=0)
sample = pd.read_csv(path+'sample_submission.csv')

# 피처와 타겟 분리
x = train.drop(['Income','Gains','Losses','Dividends','Race','Hispanic_Origin','Birth_Country','Birth_Country (Father)','Birth_Country (Mother)'], axis=1)
y = train['Income']
test = test.drop(['Gains','Losses','Dividends','Dividends','Race','Hispanic_Origin','Birth_Country','Birth_Country (Father)','Birth_Country (Mother)'], axis=1)
lb = LabelEncoder()

# print(train.columns)
# for column in train.columns:
#     print(train[column].value_counts())


# 라벨 인코딩할 열 목록
columns_to_encode = ['Gender','Education_Status','Employment_Status','Industry_Status','Occupation_Status','Martial_Status','Household_Status','Household_Summary','Citizenship','Tax_Status','Income_Status']

# 데이터프레임 x의 열에 대해 라벨 인코딩 수행
for column in columns_to_encode:
    lb.fit(x[column])
    x[column] = lb.transform(x[column])

# 데이터프레임 test_csv의 열에 대해 라벨 인코딩 수행
for column in columns_to_encode:
    lb.fit(test[column])
    test[column] = lb.transform(test[column])

# print(np.unique(x['Birth_Country (Mother)'],return_counts=True))

# 데이터 스케일링
scaler = StandardScaler()
# scaler = MinMaxScaler()

x = scaler.fit_transform(x)
test = scaler.transform(test)

import random

r = random.randint(1,500)

# 훈련 데이터와 검증 데이터 분리
x_train, x_val, y_train, y_val = train_test_split(x, y, train_size=0.9, random_state=38)

# XGBoost 모델 학습
params = {'bootstrap': True, 'max_depth': 5, 'max_features': 'sqrt', 'max_leaf_nodes': 20, 'min_impurity_decrease': 0.0, 'min_samples_leaf': 1, 'min_samples_split': 4, 'min_weight_fraction_leaf': 0, 'n_estimators': 150}


from catboost import CatBoostClassifier
# model = RandomForestClassifier()
model = CatBoostClassifier(iterations=100)
model.fit(x_train, y_train)
import joblib

# 모델 저장
joblib.dump(model, "c:/_data/dacon/soduc/weight/money_RF_1.pkl")

# 저장된 모델 불러오기
# loaded_model = joblib.load("c:/_data/_save/soduc_model.pkl")
# 검증 데이터 예측
y_pred_val = model.predict(x_val)

# 검증 데이터 RMSE 계산
rmse_val = mean_squared_error(y_val, y_pred_val, squared=False)
print("Validation RMSE:", rmse_val,'r',r)
# 테스트 데이터 예측 및 저장
y_pred_test = model.predict(test)
sample['Income'] = y_pred_test
sample.to_csv("c:/_data/dacon/soduc/csv/money_RF_1.csv", index=False)

'''
581 10번 standard random 6
minmax
Validation RMSE: 543.5683874992352 r 92 11번
Validation RMSE: 528.3234061389828 r 38 12번
'''


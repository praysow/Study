import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
import optuna
import random
import os
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

seed_everything(42)
# 데이터 불러오기
path = "c:/_data/dacon/soduc/"
train = pd.read_csv(path+'train.csv', index_col=0)
test = pd.read_csv(path+'test.csv', index_col=0)
sample = pd.read_csv(path+'sample_submission.csv')

x = train.drop(['Income','Gains','Losses','Dividends','Race','Hispanic_Origin','Birth_Country','Birth_Country (Father)','Birth_Country (Mother)'], axis=1)
y = train['Income']
test = test.drop(['Gains','Losses','Dividends','Dividends','Race','Hispanic_Origin','Birth_Country','Birth_Country (Father)','Birth_Country (Mother)'], axis=1)
lb = LabelEncoder()

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
    
# 데이터 스케일링
scaler = StandardScaler()
x = scaler.fit_transform(x)
test = scaler.transform(test)

# 훈련 데이터와 검증 데이터 분리
x_train, x_val, y_train, y_val = train_test_split(x, y, train_size=0.8, random_state=6)

model = LGBMRegressor(device = 'gpu')
model.fit(x_train,y_train)
model.booster_.save_model("c:/_data/dacon/soduc/weight/money37_lgb_optuna.csv")

# 테스트 데이터 예측 및 저장
y_pred_test = model.predict(test)
sample['Income'] = y_pred_test
sample.to_csv("c:/_data/dacon/soduc/csv/money37_lgb_optuna.csv", index=False)

y_pred_val = model.predict(x_val)
rmse_val = mean_squared_error(y_val, y_pred_val, squared=False)
print("Validation RMSE:", rmse_val)


'''
Validation RMSE: 621.3231636678684 540점
Best parameters: {'random_state': 863, 'learning_rate': 0.026448745357864258, 'n_estimators': 118, 'num_leaves': 132, 'feature_fraction': 0.9503237717724516, 'bagging_fraction': 0.22977512473974027, 'bagging_freq': 1, 'min_child_samples': 52, 'max_depth': 13, 'min_samples_leaf': 19}
Validation RMSE: 612.8145250013354      money1_lgb_optuna ???점
'''
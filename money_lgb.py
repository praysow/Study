import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import optuna

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
x_train, x_val, y_train, y_val = train_test_split(x, y, train_size=0.9, random_state=367)

def objective(trial):
    # 하이퍼파라미터 탐색 공간 정의
    lgbm_params = {
        "objective": "regression",
        "metric": "rmse",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "random_state": trial.suggest_int("random_state", 1, 1000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "num_leaves": trial.suggest_int("num_leaves", 2, 256),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.1, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.1, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        'n_jobs': -1
    }
    
    # LightGBM 모델 생성
    model = lgb.LGBMRegressor(**lgbm_params)
    
    # 모델 학습
    model.fit(x_train, y_train)
    
    # 검증 데이터 예측 및 평가
    y_pred_val = model.predict(x_val)
    rmse_val = mean_squared_error(y_val, y_pred_val, squared=False)
    
    return rmse_val

# Optuna를 사용하여 하이퍼파라미터 최적화
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=1000)

# 최적의 하이퍼파라미터 출력
best_params = study.best_params
# print('Best parameters:', best_params)

# 최적의 하이퍼파라미터로 모델 생성 및 학습
best_model = lgb.LGBMRegressor(**best_params)
best_model.fit(x_train, y_train)
best_model.booster_.save_model("c:/_data/dacon/soduc/weight/money2_lgb_optuna.csv")

# 테스트 데이터 예측 및 저장
y_pred_test = best_model.predict(test)
sample['Income'] = y_pred_test
sample.to_csv("c:/_data/dacon/soduc/csv/money2_lgb_optuna.csv", index=False)
print('Best parameters:', best_params)

y_pred_val = best_model.predict(x_val)
rmse_val = mean_squared_error(y_val, y_pred_val, squared=False)
print("Validation RMSE:", rmse_val)


'''
Validation RMSE: 621.3231636678684 540점
Best parameters: {'random_state': 863, 'learning_rate': 0.026448745357864258, 'n_estimators': 118, 'num_leaves': 132, 'feature_fraction': 0.9503237717724516, 'bagging_fraction': 0.22977512473974027, 'bagging_freq': 1, 'min_child_samples': 52, 'max_depth': 13, 'min_samples_leaf': 19}
Validation RMSE: 612.8145250013354      money1_lgb_optuna ???점
'''
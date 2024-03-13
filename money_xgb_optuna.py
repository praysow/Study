import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import optuna

# 데이터 불러오기
path = "c:/_data/dacon/soduc/"
train = pd.read_csv(path+'train.csv', index_col=0)
test = pd.read_csv(path+'test.csv', index_col=0)
sample = pd.read_csv(path+'sample_submission.csv')

# 피처와 타겟 분리
x = train.drop(['Income'], axis=1)
y = train['Income']

lb = LabelEncoder()

# 라벨 인코딩할 열 목록
columns_to_encode = ['Gender','Education_Status','Employment_Status','Industry_Status','Occupation_Status','Race','Hispanic_Origin','Martial_Status','Household_Status','Household_Summary','Citizenship','Birth_Country','Birth_Country (Father)','Birth_Country (Mother)','Tax_Status','Income_Status']

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
    xgb_params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "verbosity": 0,
        "random_state": 42,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0.0, 1.0),
        'n_jobs': -1
    }
    
    # XGBoost 모델 생성
    model = xgb.XGBRegressor(**xgb_params)
    
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

# 최적의 하이퍼파라미터로 모델 생성 및 학습
best_model = xgb.XGBRegressor(**best_params)
best_model.fit(x_train, y_train)

# 테스트 데이터 예측 및 저장
y_pred_test = best_model.predict(test)
sample['Income'] = y_pred_test
sample.to_csv("c:/_data/dacon/soduc/csv/money3_optuna_xgboost.csv", index=False)
print('Best parameters:', best_params)

y_pred_val = best_model.predict(x_val)
rmse_val = mean_squared_error(y_val, y_pred_val, squared=False)
print("Validation RMSE:", rmse_val)
'''
Validation RMSE: 617.3331620533075      1번

Validation RMSE: 617.3522769437751      2번
'''
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error
import xgboost as xgb
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
        "random_state": trial.suggest_int("random_state", 1, 1000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
        "n_estimators": trial.suggest_int("n_estimators", 1, 500),
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
    model = xgb.XGBRegressor(**xgb_params,device = 'gpu')
    
    # 모델 학습
    
    model.fit(x_train, y_train)
    
    # 검증 데이터 예측 및 평가
    y_pred_val = model.predict(x_val)
    rmse_val = mean_squared_error(y_val, y_pred_val, squared=False)
    
    return rmse_val

# Optuna를 사용하여 하이퍼파라미터 최적화
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=3000)

# 최적의 하이퍼파라미터 출력
best_params = study.best_params
import joblib
# 최적의 하이퍼파라미터로 모델 생성 및 학습
best_model = xgb.XGBRegressor(**best_params)
best_model.fit(x_train, y_train)
joblib.dump(best_model, "c:/_data/dacon/soduc/weight/money22_optuna_xgboost.pkl")
# 테스트 데이터 예측 및 저장
y_pred_test = best_model.predict(test)
sample['Income'] = y_pred_test
sample.to_csv("c:/_data/dacon/soduc/csv/money22_optuna_xgboost.csv", index=False)
print('Best parameters:', best_params)

y_pred_val = best_model.predict(x_val)
rmse_val = mean_squared_error(y_val, y_pred_val, squared=False)
print("Validation RMSE:", rmse_val)

'''
Validation RMSE: 617.3331620533075      1번

Validation RMSE: 617.3522769437751      2번

Best parameters: {'random_state': 589, 'learning_rate': 0.023765528985485895, 'n_estimators': 151, 'max_depth': 7, 'subsample': 0.9890237618912407, 'colsample_bytree': 0.7203459068696018, 'reg_alpha': 0.5062481644377009, 'reg_lambda': 0.9744913997224112, 'min_child_weight': 4, 'gamma': 0.7074286786435127}
Validation RMSE: 616.0891829434779     money4_optuna_xgboost    544점

Best parameters: {'random_state': 596, 'learning_rate': 0.01727073006198615, 'n_estimators': 191, 'max_depth': 9, 'subsample': 0.5517418635798523, 'colsample_bytree': 0.6489366724855755, 'reg_alpha': 0.7730596255445736, 'reg_lambda': 0.3018239964507463, 'min_child_weight': 9, 'gamma': 0.6048670915268969}
Validation RMSE: 615.7543318199334      money3_optuna_xgboost    542점
Validation RMSE: 614.603456139294
'''
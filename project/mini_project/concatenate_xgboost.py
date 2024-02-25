import pandas as pd
import numpy as np
import pickle
import os.path
import datetime as dt
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from preprocessing import load_bus, load_deay, load_passenger, load_weather
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import optuna
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')
import xgboost as xgb
import time
# 데이터 로드
bus_csv = load_bus()
passenger_csv = load_passenger()
weather_csv = load_weather()
delay_csv = load_deay()

# 레이블 선택
bus = bus_csv
passenger = passenger_csv
weather = weather_csv
# y = delay_csv['1호선지연(분)']
# y = delay_csv['2호선지연(분)']
y = delay_csv['3호선지연(분)']
# y = delay_csv['4호선지연(분)']
# y = delay_csv['5호선지연(분)']
# y = delay_csv['6호선지연(분)']
# y = delay_csv['7호선지연(분)']
# y = delay_csv['8호선지연(분)']

# 훈련 및 테스트 데이터 분할(원하는 상황으로 주석처리를 바꾸기)
x1_train, x1_test, x2_train, x2_test, x3_train, x3_test, y_train, y_test = train_test_split(
    bus,passenger,weather, y, train_size=0.99, random_state=100, stratify=y)
# #버스데이터로 예측
# x_train, x_test, y_train, y_test = train_test_split(bus, y, train_size=0.99, random_state=100, stratify=y)
# #인원데이터로 예측
# x_train, x_test, y_train, y_test = train_test_split(passenger, y, train_size=0.99, random_state=100, stratify=y)
# #날씨데이터로 예측
# x_train, x_test, y_train, y_test = train_test_split(weather, y, train_size=0.99, random_state=100, stratify=y)

# 스케일링(모든 데이터 이용시)
scaler1 = StandardScaler()
scaler2 = MinMaxScaler()
scaler3 = RobustScaler()
# scaler = MaxAbsScaler()
x1_train_scaled = scaler1.fit_transform(x1_train)
x1_test_scaled = scaler1.transform(x1_test)

x2_train_scaled = scaler2.fit_transform(x2_train)
x2_test_scaled = scaler2.transform(x2_test)

x3_train_scaled = scaler3.fit_transform(x3_train)
x3_test_scaled = scaler3.transform(x3_test)
# 스케일링(각각)
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)
# 데이터 연결(모든 데이터 이용시)
x_train = np.concatenate((x1_train_scaled, x2_train_scaled, x3_train_scaled), axis=1)
x_test = np.concatenate((x1_test_scaled, x2_test_scaled, x3_test_scaled), axis=1)
s_t = time.time()
def objective(trial):
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'random_state': 67,
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),
        'n_estimators': trial.suggest_int('n_estimators', 50, 200),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'min_child_weight': trial.suggest_loguniform('min_child_weight', 1e-8, 100.0),
        'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_loguniform('gamma', 1e-8, 100.0),
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-2, 100.0),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-2, 100.0),
    }

    model = XGBRegressor(**params, tree_method='gpu_hist')
    model.fit(x_train, y_train,
              eval_set=[(x_test, y_test)],
              early_stopping_rounds=1000,
              verbose=False)
    
    y_pred = model.predict(x_test)
    
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    print("RMSE:", rmse)
    
    r2 = r2_score(y_test, y_pred)
    print("R2 Score:", r2)
    
    return r2

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=10000)
#n_trials : 최적화를 위해 시도할 하이퍼파라미터 조합의 수
e_t = time.time()
print("Best parameters found: ", study.best_params)
print("R2: ", study.best_value)
print("시간",e_t-s_t)
# 모델 저장이 잘 안되서 사용금지
# 모델 저장
model_path = 'c:/_data/_save/project/mini_project_xgb.pkl'
params = study.best_params
model = XGBRegressor(**params, tree_method='gpu_hist')
model.fit(x_train, y_train)
with open(model_path, 'wb') as f:
    pickle.dump(model, f)

    
    

'''
 Trial 9999 finished with value: 0.5741039298393602 and parameters: {'learning_rate': 0.2994094719224554, 'n_estimators': 144, 'max_depth': 3, 'min_child_weight': 0.0817367384228868, 'subsample': 0.5314912903048394, 'colsample_bytree': 0.6615345838950986, 'gamma': 4.8361268098439245, 'reg_alpha': 0.29105262353638695, 'reg_lambda': 0.026092007573406684}. Best is trial 8985 with value: 0.9417521287221309.
Best parameters found:  {'learning_rate': 0.2820565332069652, 'n_estimators': 86, 'max_depth': 3, 'min_child_weight': 2.7182174248831177, 'subsample': 0.5412259534520243, 'colsample_bytree': 0.6877608294947486, 'gamma': 0.20588878392300902, 'reg_alpha': 1.3058471358851158, 'reg_lambda': 0.12068888586502122}
R2:  0.9417521287221309
'''

#######################################파라미터 설명########################################
# objective: 손실 함수를 지정합니다. 여기서는 회귀 문제를 다루고 있으므로 'reg:squarederror'를 사용하여 평균 제곱 오차를 최소화합니다.

# eval_metric: 모델을 평가하는 데 사용되는 지표를 지정합니다. 여기서는 평균 제곱근 오차(RMSE)를 사용합니다.

# random_state: 모델의 랜덤 시드를 설정합니다.

# learning_rate: 각 트리의 가중치를 줄이는 데 사용되는 학습 속도를 나타냅니다.

# n_estimators: 부스팅 라운드의 수, 즉 트리의 개수를 나타냅니다.

# max_depth: 트리의 최대 깊이를 제한합니다.

# min_child_weight: 리프 노드에 필요한 최소 가중치 합을 나타냅니다.

# subsample: 트리를 학습하는 데 사용되는 데이터의 일부분을 나타냅니다.

# colsample_bytree: 각 트리를 학습할 때 사용되는 특성의 비율을 나타냅니다.

# gamma: 트리의 분할을 결정하는 데 사용되는 최소 손실 감소를 나타냅니다.

# reg_alpha: L1 정규화를 제어하는 매개변수입니다.

# reg_lambda: L2 정규화를 제어하는 매개변수입니다.
import pandas as pd
import numpy as np
import pickle
import os.path
import datetime as dt
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler,RobustScaler
from preprocessing import load_bus,load_passenger, load_weather,load_deay
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error
import optuna
import lightgbm as lgb
from keras.layers import concatenate
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')
# 데이터 로드
# bus_csv = load_bus()
# passenger_csv = load_passenger()
# weather_csv = load_weather()
# delay_csv = load_deay()

passenger_csv = pd.read_csv("c:/mini_project/data/passenger_sujong.csv")
weather_csv = pd.read_csv("c:/mini_project/data/weather_sujong.csv")
delay_csv = pd.read_csv("c:/mini_project/data/delay_sujong.csv")
bus_csv = pd.read_csv("c:/mini_project/data/bus_sujong.csv")

passenger_csv = passenger_csv.rename(columns={'Unnamed: 0':'일시'})
delay_csv = delay_csv.rename(columns={'Unnamed: 0':'일시'})
bus_csv = bus_csv.rename(columns={'Unnamed: 0':'일시'})


# print(delay_csv.columns)    #['1호선지연(분)', '2호선지연(분)', '3호선지연(분)', '4호선지연(분)', '5호선지연(분)', '6호선지연(분)',
    #    '7호선지연(분)', '8호선지연(분)']

# 레이블 선택
x1 = bus_csv
x2 = passenger_csv
x3 = weather_csv
y_col = delay_csv['1호선지연(분)']

size = 30
pred_step = 2

def split_xy(data, time_step, y_col, pred_step):
    result_x = []
    result_y = []
    
    num = len(data) - (time_step + pred_step)
    for i in range(num):
        result_x.append(data[i:i+time_step])
        result_y.append(data.iloc[i+time_step+pred_step][y_col])
        
    return np.array(result_x), np.array(result_y)  # 리스트를 numpy 배열로 변환하여 반환

 # y_col 인자 대신 data 자체를 선택하여 데이터를 가져옵니다
x1, y = split_xy(x1, size, pred_step)
x2, y = split_xy(x2, size, '1호선지연(분)', pred_step)
x3, y = split_xy(x3, size, '1호선지연(분)', pred_step)

print(x1.shape)
print(x2.shape)
print(x3.shape)

x1 = x1.reshape(5800,30*2)
x2 = x2.reshape(5800,282*30)
x3 = x3.reshape(5800,30*3)

print(y.shape)

'''

# 훈련 및 테스트 데이터 분할
x1_train, x1_test, x2_train, x2_test, x3_train, x3_test, y_train, y_test = train_test_split(
    x1, x2, x3, y, train_size=0.99, random_state=100,shuffle=True,stratify=y
)

# 스케일링
scaler = StandardScaler()
x1_train_scaled = scaler.fit_transform(x1_train)
x1_test_scaled = scaler.transform(x1_test)

scaler = MinMaxScaler()
x2_train_scaled = scaler.fit_transform(x2_train)
x2_test_scaled = scaler.transform(x2_test)

scaler = RobustScaler()
x3_train_scaled = scaler.fit_transform(x3_train)
x3_test_scaled = scaler.transform(x3_test)

# 데이터 연결
x_train = np.concatenate((x1_train_scaled, x2_train_scaled, x3_train_scaled), axis=1)
x_test = np.concatenate((x1_test_scaled, x2_test_scaled, x3_test_scaled), axis=1)

def objective(trial):
    params = {
        "metric": "mse",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "random_state": 67,
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),
        'n_estimators': trial.suggest_int('n_estimators', 50, 200),
        'num_leaves': trial.suggest_int('num_leaves', 10, 50),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 30),
        'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-2, 100.0),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-2, 100.0),
        'min_split_gain': trial.suggest_loguniform('min_split_gain', 1e-8, 100.0),
        'min_child_weight': trial.suggest_loguniform('min_child_weight', 1e-8, 100.0),
        'cat_smooth': trial.suggest_int('cat_smooth', 1, 100),
        "early_stopping_rounds": 1000,  # 얼리 스탑
    }
    
    model = lgb.LGBMRegressor(**params, device='gpu')
    
    model.fit(x_train, y_train,
              eval_set=[(x_test, y_test)],
              eval_metric='mse',)
    
    y_pred = model.predict(x_test)
    
    mse = mean_squared_error(y_test, y_pred)
    print("MSE:", mse)
    
    r2 = r2_score(y_test, y_pred)
    print("R^2 Score:", r2)
    
    return r2

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)

print("Best parameters found: ", study.best_params)
print("r2: ", study.best_value)
# best_params = study.best_params
# best_model = lgb.LGBMClassifier(**best_params, device='gpu')
# best_model.fit(x_train, y_train)
# best_model.booster_.save_model("c:/_data/_save/project/mini_project_lightbgm.h5")
'''
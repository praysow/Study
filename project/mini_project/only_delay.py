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
# 데이터 로드
bus_csv = load_bus()
passenger_csv = load_passenger()
weather_csv = load_weather()
delay_csv = load_deay()



# print(delay_csv.columns)    #['1호선지연(분)', '2호선지연(분)', '3호선지연(분)', '4호선지연(분)', '5호선지연(분)', '6호선지연(분)',
    #    '7호선지연(분)', '8호선지연(분)']

# 레이블 선택
x= delay_csv
y = delay_csv['1호선지연(분)']
# y = delay_csv['2호선지연(분)']
# y = delay_csv['3호선지연(분)']
# y = delay_csv['4호선지연(분)']
# y = delay_csv['5호선지연(분)']
# y = delay_csv['6호선지연(분)']
# y = delay_csv['7호선지연(분)']
# y = delay_csv['8호선지연(분)']
#몇개씩 묶어서 훈련을 할 것인가
size = 30
#몇개의 데이터 뒤를(시간,날짜 등) 예측할것인가
pred_step = 2
def split_xy(data, time_step, y, pred_step):
    result_x = []
    result_y = []
    
    num = len(data) - (time_step + pred_step)
    for i in range(num):
        result_x.append(data[i:i+time_step])
        result_y.append(data.iloc[i+time_step+pred_step][y])
        
    return np.array(result_x), np.array(result_y)  # 리스트를 numpy 배열로 변환하여 반환

 # y_col 인자 대신 data 자체를 선택하여 데이터를 가져옵니다
x, y = split_xy(x, size,'1호선지연(분)', pred_step)

# print(x.shape)
x = x.reshape(5800,size*8)
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.99, random_state=100, stratify=y)


# scaler = StandardScaler()
# scaler = MinMaxScaler()
scaler = RobustScaler()
# scaler = MaxAbsScaler()

# 스케일링
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

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
study.optimize(objective, n_trials=100)
#n_trials : 최적화를 위해 시도할 하이퍼파라미터 조합의 수

print("Best parameters found: ", study.best_params)
print("R2: ", study.best_value)

'''
standard
R2:  0.6555876320215899
minmax
R2:  0.6552108809848036
Robust
R2:  0.6667764776232703
maxabs
R2:  0.6569345332434289
파라미터를 고정시키진 않았지만 대체적으로 비슷하게 R2값이 나옴
'''

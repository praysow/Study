import pandas as pd
import numpy as np
import pickle
import os.path
import datetime as dt
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler,RobustScaler
from preprocessing import load_bus, load_deay, load_passenger, load_weather
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import optuna
from keras.layers import concatenate
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')
from catboost import CatBoostRegressor
# 데이터 로드
bus_csv = load_bus()
passenger_csv = load_passenger()
weather_csv = load_weather()
delay_csv = load_deay()

# 레이블 선택
bus = bus_csv
passenger = passenger_csv
weather = weather_csv
y = delay_csv['1호선지연(분)']
# y = delay_csv['2호선지연(분)']
# y = delay_csv['3호선지연(분)']
# y = delay_csv['4호선지연(분)']
# y = delay_csv['5호선지연(분)']
# y = delay_csv['6호선지연(분)']
# y = delay_csv['7호선지연(분)']
# y = delay_csv['8호선지연(분)']

# 훈련 및 테스트 데이터 분할(원하는 상황으로 주석처리를 바꾸기)
# x1_train, x1_test, x2_train, x2_test, x3_train, x3_test, y_train, y_test = train_test_split(
#     bus,passenger,weather, y, train_size=0.99, random_state=100, stratify=y)
# #버스데이터로 예측
x_train, x_test, y_train, y_test = train_test_split(bus, y, train_size=0.99, random_state=100, stratify=y)
# #인원데이터로 예측
# x_train, x_test, y_train, y_test = train_test_split(passenger, y, train_size=0.99, random_state=100, stratify=y)
# #날씨데이터로 예측
# x_train, x_test, y_train, y_test = train_test_split(weather, y, train_size=0.99, random_state=100, stratify=y)

# 스케일링(모든 데이터 이용시)
scaler = StandardScaler()
# scaler = MinMaxScaler()
# scaler = RobustScaler()
# scaler = MaxAbsScaler()
# x1_train_scaled = scaler.fit_transform(x1_train)
# x1_test_scaled = scaler.transform(x1_test)

# x2_train_scaled = scaler.fit_transform(x2_train)
# x2_test_scaled = scaler.transform(x2_test)

# x3_train_scaled = scaler.fit_transform(x3_train)
# x3_test_scaled = scaler.transform(x3_test)
# 스케일링(각각)
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
# 데이터 연결(모든 데이터 이용시)
# x_train = np.concatenate((x1_train_scaled, x2_train_scaled, x3_train_scaled), axis=1)
# x_test = np.concatenate((x1_test_scaled, x2_test_scaled, x3_test_scaled), axis=1)

def objective(trial):
    # 하이퍼파라미터 탐색 공간 정의
    params = {
        'iterations': trial.suggest_int('iterations', 150, 300),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),
        'depth': trial.suggest_int('depth', 4, 10),
        'l2_leaf_reg': trial.suggest_loguniform('l2_leaf_reg', 1e-8, 10.0),
        'bagging_temperature': trial.suggest_loguniform('bagging_temperature', 1e-7, 10.0),
        'random_strength': trial.suggest_loguniform('random_strength', 1e-8, 10.0),
        'border_count': trial.suggest_int('border_count', 170, 350),
        'verbose': False,
        }

    # CatBoost 모델 생성(회귀)
    model = CatBoostRegressor(**params,devices='gpu')
    
    # 모델 학습
    model.fit(x_train, y_train,eval_set=[(x_test, y_test)],)
    
    y_pred = model.predict(x_test)
    
    mse = mean_squared_error(y_test, y_pred)
    print("MSE:", mse)
    
    r2 = r2_score(y_test, y_pred)
    print("R2 Score:", r2)
    
    return r2
#optuna를 사용해서 최적의 파라미터 찾기
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)

print("Best parameters found: ", study.best_params)
print("r2: ", study.best_value)

# 최적의 모델 생성
# best_params = study.best_params
# best_model = CatBoostRegressor(**best_params,devices = 'gpu')

# # 최적의 모델 학습
# best_model.fit(x_train, y_train, eval_set=(x_test, y_test), verbose=False)

# # 모델 저장
# best_model.save_model("c:/_data/_save/project/mini_project_catboost..bin')
'''
Trial 99 finished with value: 0.3834932053464405 and parameters: {'iterations': 181, 'learning_rate': 0.01085648772400954, 'depth': 5, 'l2_leaf_reg': 0.15014931222078134, 'bagging_temperature': 9.924972743578977, 'random_strength': 6.524818007153559e-07, 'border_count': 188}. Best is trial 74 with value: 0.38028029524711415.
Best parameters found:  {'iterations': 160, m 'learning_rate': 0.010007590675058442, 'depth': 5, 'l2_leaf_reg': 1.7642249481588387e-07, 'bagging_temperature': 1.8981025511571716, 'random_strength': 1.974135970216598e-06, 'border_count': 218}
r2:  0.38028029524711415
'''
############################ 파라미터 설명  ##############################
# iterations: 부스팅 라운드의 수를 나타냅니다. 부스팅 라운드는 모델이 학습 데이터를 반복해서 학습하는 횟수를 말합니다. 
#             이 매개변수는 학습 과정의 반복 횟수를 지정합니다. 일반적으로 너무 낮거나 높으면 과소적합 또는 과적합을 일으킬 수 있습니다.

# learning_rate: 각 부스팅 단계에서 적용되는 학습 속도를 나타냅니다. 학습률은 각 트리가 이전 트리에서 얼마나 많은 오류를 수정해야 하는지를 결정합니다. 
#                보통 0.01에서 0.3 사이의 값을 가집니다.

# depth: 트리의 최대 깊이를 나타냅니다. 각 트리의 최대 깊이를 제어하여 모델의 복잡성을 조절할 수 있습니다. 너무 깊은 트리는 과적합을 야기할 수 있습니다.

# l2_leaf_reg: L2 정규화 계수를 나타냅니다. 이것은 L2 규제를 통해 가중치를 규제하는 데 사용됩니다. 모델의 복잡성을 줄이고 과적합을 방지하는 데 도움이 됩니다.

# bagging_temperature: 샘플링을 수행할 때 사용되는 베깅 온도를 나타냅니다. 이 매개변수는 데이터 샘플의 가중치를 조절하여 모델의 일반화 성능을 향상시키는 데 도움을 줍니다.

# random_strength: 랜덤성을 제어하는 매개변수로, 모델이 다양한 특징을 학습하도록 도와줍니다.

# border_count: 특성 값의 히스토그램을 구성할 때 사용되는 이진 트리의 잎 노드의 최대 개수를 나타냅니다. 이 매개변수는 모델의 효율성을 조절하는 데 사용됩니다

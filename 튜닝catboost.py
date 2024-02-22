import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier, Pool, cv
import optuna

path = "c:/_data/kaggle/비만/"
train = pd.read_csv(path + "train.csv", index_col=0)
test = pd.read_csv(path + "test.csv", index_col=0)
sample = pd.read_csv(path + "sample_submission.csv")
x = train.drop(['NObeyesdad'], axis=1)
y = train['NObeyesdad']

lb = LabelEncoder()

# 라벨 인코딩할 열 목록
columns_to_encode = ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS']

# 데이터프레임 x의 열에 대해 라벨 인코딩 수행
for column in columns_to_encode:
    lb.fit(x[column])
    x[column] = lb.transform(x[column])

# 데이터프레임 test_csv의 열에 대해 라벨 인코딩 수행
for column in columns_to_encode:
    lb.fit(test[column])
    test[column] = lb.transform(test[column])

X_train, X_valid, y_train, y_valid = train_test_split(x, y, test_size=0.2, random_state=42)

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
        'loss_function': 'MultiClass',
        'verbose': False,
        }
    
    # CatBoost 모델 생성
    model = CatBoostClassifier(**params)
    
    # 모델 학습
    model.fit(X_train, y_train, eval_set=(X_valid, y_valid), verbose=False)
    
    # 검증 데이터에 대한 예측 수행
    y_pred = model.predict(X_valid)
    
    # 정확도 계산
    accuracy = accuracy_score(y_valid, y_pred)
    
    return accuracy  # 정확도를 최대화하기 위해 1-accuracy를 반환

# Study 생성 및 최적화 수행
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)

# 최적의 하이퍼파라미터 출력
print("Best parameters found: ", study.best_params)
print("Best value found: ", study.best_value)

# 최적의 모델 생성
best_params = study.best_params
best_model = CatBoostClassifier(**best_params,devices = 'gpu')

# 최적의 모델 학습
best_model.fit(X_train, y_train, eval_set=(X_valid, y_valid), verbose=False)

# 모델 저장
best_model.save_model('catboost_model.bin')

# 테스트 데이터 예측
# 테스트 데이터 예측
y_submit = best_model.predict(test)

# y_submit이 2차원 배열인 경우 1차원 배열로 변환
if y_submit.ndim == 2:
    y_submit = np.squeeze(y_submit)


# 결과를 CSV 파일에 저장
sample['NObeyesdad'] = y_submit
sample.to_csv(path + '비만cat.csv', index=False)


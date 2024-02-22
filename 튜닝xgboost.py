import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import accuracy_score
import xgboost as xgb
import optuna
import pickle

path = "c:/_data/kaggle/비만/"
train = pd.read_csv(path + "train.csv", index_col=0)
test = pd.read_csv(path + "test.csv", index_col=0)
sample = pd.read_csv(path + "sample_submission.csv")
x = train.drop(['NObeyesdad'], axis=1)
y = train['NObeyesdad']

# 합쳐서 라벨 인코딩 수행
combined = pd.concat([x, test], axis=0)

lb = LabelEncoder()
ohe = OneHotEncoder(sparse=False)

# 라벨 인코딩할 열 목록
columns_to_encode = ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS']

for column in columns_to_encode:
    lb.fit(combined[column])
    combined[column] = lb.transform(combined[column])

# 데이터 다시 분할
x = combined[:len(train)]
test = combined[len(train):]

y = lb.fit_transform(y)  # 클래스 라벨 인코딩
y_ohe = ohe.fit_transform(y.reshape(-1, 1))  # 클래스 원-핫 인코딩

X_train, X_valid, y_train, y_valid = train_test_split(x, y_ohe, test_size=0.2, random_state=42)

dtrain = xgb.DMatrix(X_train, label=y_train)
dvalid = xgb.DMatrix(X_valid, label=y_valid)

def objective(trial):
    params = {
        'objective': 'multi:softmax',  # 목적 함수 설정
        'num_class': len(np.unique(y)),  # 클래스 수
        'eval_metric': 'merror',
        'verbosity': 0,
        'booster': 'gbtree',
        'lambda': trial.suggest_loguniform('lambda', 1e-8, 1.0),
        'alpha': trial.suggest_loguniform('alpha', 1e-8, 1.0),
        'max_depth': trial.suggest_int('max_depth', 3, 9),
        'eta': trial.suggest_loguniform('eta', 0.01, 0.3),
        'gamma': trial.suggest_loguniform('gamma', 1e-8, 1.0),
        'grow_policy': 'depthwise',
        'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),
    }

    model = xgb.train(params, dtrain)
    
    y_pred = model.predict(dvalid)
    
    accuracy = accuracy_score(y_valid, y_pred)
    
    return 1.0 - accuracy

# Study 생성 및 최적화 수행
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)

# 최적의 하이퍼파라미터 출력
print("Best parameters found: ", study.best_params)
print("Best value found: ", study.best_value)

# 최적의 하이퍼파라미터로 모델 재학습
best_params = study.best_params
best_params['objective'] = 'multi:softmax'
best_params['num_class'] = len(np.unique(y))
best_model = xgb.train(best_params, dtrain)

# 모델 저장
with open('xgboost_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

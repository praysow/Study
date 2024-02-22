import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import lightgbm as lgb
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
    x[column] = lb.fit_transform(x[column])

# 데이터프레임 test_csv의 열에 대해 라벨 인코딩 수행
for column in columns_to_encode:
    test[column] = lb.transform(test[column])

X_train, X_valid, y_train, y_valid = train_test_split(x, y, test_size=0.2, random_state=367)

def objective(trial):
    params = {
        "objective": "multiclass",
        "metric": "multi_logloss",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "random_state": 67,
        "num_class": 7,
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.1),
        'n_estimators': trial.suggest_int('n_estimators', 50, 100),
        'num_leaves': trial.suggest_int('num_leaves', 5, 50),
        'max_depth': trial.suggest_int('max_depth', 3, 7),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
        'subsample': trial.suggest_uniform('subsample', 0.5, 0.9),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 0.9),
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-8, 100.0),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-8, 100.0),
        'min_split_gain': trial.suggest_loguniform('min_split_gain', 1e-8, 1.0),
        'min_child_weight': trial.suggest_loguniform('min_child_weight', 1e-8, 100.0),
        'cat_smooth': trial.suggest_int('cat_smooth', 1, 20),
        "early_stopping_rounds": 10,  # 얼리 스탑
    }
    
    model = lgb.LGBMClassifier(**params, device='gpu')
    
    model.fit(X_train, y_train,
              eval_set=[(X_valid, y_valid)],
              eval_metric='multi_error',)
    
    y_pred = model.predict(X_valid)
    
    accuracy = accuracy_score(y_valid, y_pred)
    
    return 1.0 - accuracy

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)

print("Best parameters found: ", study.best_params)
print("Best value found: ", study.best_value)

best_params = study.best_p

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
import lightgbm as lgb
import optuna

path= "c:/_data/kaggle/비만/"
train=pd.read_csv(path+"train.csv",index_col=0)
test=pd.read_csv(path+"test.csv",index_col=0)
sample=pd.read_csv(path+"sample_submission.csv")
x= train.drop(['NObeyesdad'],axis=1)
y= train['NObeyesdad']

lb = LabelEncoder()

columns_to_encode = ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS']

for column in columns_to_encode:
    lb.fit(x[column])
    x[column] = lb.transform(x[column])

for column in columns_to_encode:
    lb.fit(test[column])
    test[column] = lb.transform(test[column])

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, random_state=367, stratify=y, shuffle=True)

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
test = scaler.transform(test)

def objective(trial):
    params = {
        'objective': 'multiclass',
        'metric': 'multi_logloss',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'random_state': 67,
        'num_class': 7,
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'num_leaves': trial.suggest_int('num_leaves', 10, 50),
        'max_depth': trial.suggest_int('max_depth', 5, 20),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 30),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-6, 1e3, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-6, 1e3, log=True),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
        'min_child_weight': trial.suggest_float('min_child_weight', 1e-6, 1e3, log=True),
    }

    model = lgb.LGBMClassifier(**params)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("acc",accuracy)
    return accuracy

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=500)

best_params = study.best_params
best_model = lgb.LGBMClassifier(**best_params)
best_model.fit(x_train, y_train)
best_model.booster_.save_model("c:/_data/_save/비만58.h5")
y_pred = best_model.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
# 테스트 데이터 예측 및 저장
y_submit = best_model.predict(test)
sample['NObeyesdad'] = y_submit
sample.to_csv(path + "비만58_optuna.csv", index=False)
print("Best Params:", best_params)
print("Test Accuracy:", accuracy)

'''
Best Params: {'learning_rate': 0.01273757550090559, 'n_estimators': 853, 'num_leaves': 21, 'max_depth': 18, 'min_child_samples': 16, 'reg_alpha': 0.0006685778456631791, 'reg_lambda': 0.00035520280512296894, 'feature_fraction': 0.5257498786717125, 'bagging_fraction': 0.705923516322051, 'bagging_freq': 1, 'min_child_weight': 0.00024132621423189556}
Test Accuracy: 0.924373795761079    56번
'''

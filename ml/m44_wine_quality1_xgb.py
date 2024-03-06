import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
import optuna
path = 'c:/_data/dacon/wine/'
train = pd.read_csv(path + 'train.csv', index_col=0)
test = pd.read_csv(path + 'test.csv', index_col=0)
submit = pd.read_csv(path + 'sample_submission.csv')

x = train.drop(['quality'], axis=1)
y = train['quality']
y -= 3
lb = LabelEncoder()
lb.fit(x['type'])
x['type'] = lb.transform(x['type'])
test['type'] = lb.transform(test['type'])

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, random_state=8)

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

def objective(trial):
    params = {
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

    model = XGBClassifier(**params,device='cuda')
    model.fit(x_train, y_train,
              eval_set=[(x_test, y_test)],
              early_stopping_rounds=20,
              verbose=False)
    
    y_pred = model.predict(x_test)
    
    r2 = accuracy_score(y_test, y_pred)
    print("R2 Score:", r2)
    
    return r2

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)
#n_trials : 최적화를 위해 시도할 하이퍼파라미터 조합의 수
print("Best parameters found: ", study.best_params)
print("R2: ", study.best_value)
# 모델 저장이 잘 안되서 사용금지
# 모델 저장
# model_path = 'c:/_data/_save/project/mini_project_xgb_8호선.pkl'
# params = study.best_params
# model = XGBclassifier(**params, tree_method='gpu_hist')
# model.fit(x_train, y_train)
# with open(model_path, 'wb') as f:
#     pickle.dump(model, f)

'''
Best parameters found:  {'learning_rate': 0.06884291104115536, 'n_estimators': 177, 'max_depth': 8, 'min_child_weight': 0.0025489501603337345, 'subsample': 0.700296122199031, 'colsample_bytree': 0.6340324994107785, 'gamma': 1.0217337541516813e-06, 'reg_alpha': 0.5435807446717495, 'reg_lambda': 0.05886566315693689}
R2:  0.7327272727272728
'''
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,RobustScaler,MinMaxScaler,MaxAbsScaler
from sklearn.metrics import roc_auc_score
import optuna

# 데이터 불러오기
data = pd.read_csv('c:/_data/dacon/ranfo/ts/train.csv')
submit = pd.read_csv('c:/_data/dacon/ranfo/ts/sample_submission.csv')

# person_id 컬럼 제거
x = data.drop(['person_id', 'login'], axis=1)
y = data['login']

# 데이터 분할
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, random_state=6)

# 데이터 스케일링
scaler = StandardScaler()
# scaler = RobustScaler()
# scaler = MinMaxScaler()
# scaler = MaxAbsScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# 이전 시험의 최적 성능을 저장하기 위한 변수
best_auc = float('-inf')

def objective(trial):
    global best_auc
    
    # 하이퍼파라미터 탐색 공간 정의
    n_estimators = trial.suggest_int('n_estimators', 10, 1000)
    max_depth = trial.suggest_int('max_depth', 3, 30)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 20)
    min_weight_fraction_leaf = trial.suggest_float('min_weight_fraction_leaf', 0.0, 0.5)
    max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
    bootstrap = trial.suggest_categorical('bootstrap', [True, False])
    ccp_alpha = trial.suggest_float('ccp_alpha', 0.0, 1.0)
    
    # 모델 생성
    model = RandomForestClassifier(
        n_estimators=n_estimators, 
        max_depth=max_depth, 
        min_samples_split=min_samples_split, 
        min_samples_leaf=min_samples_leaf, 
        max_features=max_features, 
        bootstrap=bootstrap,
        ccp_alpha=ccp_alpha,
         min_weight_fraction_leaf =  min_weight_fraction_leaf,
        random_state=42
    )

    # 모델 학습
    model.fit(x_train, y_train)
    # model.fit(x, y)

    # 검증 세트에서의 AUC 계산
    y_pred_proba = model.predict_proba(x_test)[:, 1]
    # y_pred_proba = model.predict_proba(x)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)
    # auc = roc_auc_score(y, y_pred_proba)
    
    # 이전 시험보다 성능이 좋으면 최적 성능 및 파라미터 업데이트
    if auc > best_auc:
        best_auc = auc
    
    return auc


# Optuna를 사용하여 하이퍼파라미터 최적화
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=500)

# 최적의 하이퍼파라미터 및 AUC 출력
best_params = study.best_params
best_auc = study.best_value
print('Best parameters:', best_params)
print('Best AUC:', best_auc)

# 찾은 최적의 파라미터들을 제출 양식에 맞게 제출
for param, value in best_params.items():
    if param in submit.columns:
        submit[param] = value

submit.to_csv('c:/_data/dacon/ranfo/rf_optuna_8.csv', index=False)


'''
Best parameters: {'n_estimators': 840, 'max_depth': 12, 'min_samples_split': 5, 'min_samples_leaf': 5, 'max_features': 'log2', 'bootstrap': True, 'ccp_alpha': 0.012083375663032386}
Best AUC: 0.8360805860805861        2번 train_test_split o      0.81점

Best parameters: {'n_estimators': 124, 'max_depth': 29, 'min_samples_split': 9, 'min_samples_leaf': 14, 'max_features': None, 'bootstrap': True, 'ccp_alpha': 0.00021645360278310655}
Best AUC: 0.8777046843896865        3번 train_test_split x      0.80점

Best parameters: {'n_estimators': 515, 'max_depth': 7, 'min_samples_split': 4, 'min_samples_leaf': 11, 'min_weight_fraction_leaf': 0.010964274165023224, 'max_features': 'log2', 'bootstrap': True, 'ccp_alpha': 5.461459086784141e-05}
Best AUC: 0.8914602346805737        3번 train_test_split o      0.19점

Best parameters: {'n_estimators': 106, 'max_depth': 26, 'min_samples_split': 17, 'min_samples_leaf': 14, 'min_weight_fraction_leaf': 0.033069777807453, 'max_features': 'sqrt', 'bootstrap': True, 'ccp_alpha': 0.01857077758038466}
Best AUC: 0.8492063492063492        8번 train_test_split o 
'''
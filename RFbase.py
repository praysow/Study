import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import StandardScaler
data = pd.read_csv('c:/_data/dacon/ranfo/ts/train.csv')
submit = pd.read_csv('c:/_data/dacon/ranfo/ts/sample_submission.csv')
# person_id 컬럼 제거
x = data.drop(['person_id', 'login'], axis=1)
y = data['login']

# GridSearchCV를 위한 하이퍼파라미터 설정
param_search_space = {
    'n_estimators': [100, 150],
    'max_depth': [30,50],
    'min_samples_split': [2,4],
    'min_samples_leaf': [1, 4],
    'min_weight_fraction_leaf' : [0,0.5],
    'max_features' : ['sqrt', 'log2'],
    'max_leaf_nodes' : [30, 50],
    'min_impurity_decrease' : [ 0.5],
    'bootstrap' : [True, False]
}

# RandomForestClassifier 객체 생성
rf = RandomForestClassifier(random_state=42)

# GridSearchCV 객체 생성
model = GridSearchCV(estimator=rf, param_grid=param_search_space, n_jobs=-1, verbose=2, scoring='roc_auc')

# GridSearchCV를 사용한 학습
model.fit(x, y)

# 최적의 파라미터와 최고 점수 출력
best_params = model.best_params_
best_score = model.best_score_

best_params, best_score
print('params',best_params)
print('score',best_score)

# # 찾은 최적의 파라미터들을 제출 양식에 맞게 제출
for param, value in best_params.items():
    if param in submit.columns:
        submit[param] = value

submit.to_csv('c:/_data/dacon/RF14.csv', index=False)
'''
0.84 10번

params {'bootstrap': True, 'max_depth': None, 'max_features': 'sqrt', 'max_leaf_nodes': 10, 'min_impurity_decrease': 0.0, 'min_samples_leaf': 2, 'min_samples_split': 10, 'min_weight_fraction_leaf': 0, 'n_estimators': 100}
score 0.8293217740138367 11번

params {'bootstrap': True, 'max_depth': None, 'max_features': 'sqrt', 'max_leaf_nodes': 10, 'min_impurity_decrease': 0.0, 'min_samples_leaf': 4, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0, 'n_estimators': 250}
score 0.8173551055610273    12번
'''
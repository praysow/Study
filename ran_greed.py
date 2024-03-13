import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import StandardScaler
data = pd.read_csv('c:/_data/dacon/ranfo/train.csv')
submit = pd.read_csv('c:/_data/dacon/ranfo/sample_submission.csv')
# person_id 컬럼 제거
x = data.drop(['person_id', 'login'], axis=1)
y = data['login']

x_train,x_test,y_train,y_test=train_test_split(x,y, train_size=0.8, random_state=6)

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# GridSearchCV를 위한 하이퍼파라미터 설정
param_search_space = {
    'n_estimators': [10, 50, 100],
    'max_depth': [None, 10, 30],
    'min_samples_split': [2, 10],
    'min_samples_leaf': [1, 4]
}

# RandomForestClassifier 객체 생성
rf = RandomForestClassifier(random_state=42)

# GridSearchCV 객체 생성
model = GridSearchCV(estimator=rf, param_grid=param_search_space, cv=3, n_jobs=-1, verbose=2, scoring='roc_auc')

# GridSearchCV를 사용한 학습
model.fit(x_train, y_train)

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

submit.to_csv('c:/_data/dacon/ranfo/baseline_submit.csv', index=False)


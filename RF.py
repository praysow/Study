import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import pickle
# 데이터 불러오기
data = pd.read_csv('c:/_data/dacon/ranfo/train.csv')
submit = pd.read_csv('c:/_data/dacon/ranfo/sample_submission.csv')

# 불필요한 열 제거
x = data.drop(['person_id', 'login'], axis=1)
y = data['login']

import random

r = random.randint(1,500)
# 데이터 분할
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=r)

# 특성 표준화
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# RandomForestClassifier의 기본 파라미터 가져오기
default_params = RandomForestClassifier().get_params()

# submit 데이터프레임에 기본 파라미터 저장
for param, value in default_params.items():
    if param in submit.columns:
        submit[param] = value

path = 'c:/_data/dacon/ranfo/default7'
# CSV 파일로 submit 데이터프레임 저장
submit.to_csv(path +'.csv', index=False)

# RandomForestClassifier 객체 생성 (기본 매개변수 사용)
rf_classifier = RandomForestClassifier()

with open(path + ".pkl", "wb") as f:
    pickle.dump(rf_classifier, f)

# 모델 학습
rf_classifier.fit(x_train, y_train)

# 테스트 데이터에 대한 예측 확률
y_pred_proba = rf_classifier.predict_proba(x_test)[:, 1]

# ROC AUC 계산
auc_score = roc_auc_score(y_test, y_pred_proba)
print("AUC 값:", auc_score)
print('r',r)

'''
AUC 값: 0.8223039215686274  0번
AUC 값: 0.8352591036414566  2번
AUC 값: 0.836046918767507   3번
AUC 값: 0.8604139504563234  r 263   4번
AUC 값: 0.8578059071729958  r 419   5번
AUC 값: 0.8843101343101343  r 57    6번
'''


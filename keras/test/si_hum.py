import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# 데이터 불러오기
# 데이터 불러오기
path = "c:/_data/kaggle/비만/"
train = pd.read_csv(path + "train.csv", index_col=0)
test = pd.read_csv(path + "test.csv", index_col=0)
sample = pd.read_csv(path + "sample_submission.csv")

# LabelEncoder를 사용하여 'NObeyesdad' 열을 숫자로 변환
le = LabelEncoder()
train['NObeyesdad'] = le.fit_transform(train['NObeyesdad'])

# 특성과 라벨 분리
x = train.drop(['NObeyesdad'], axis=1)
y = train['NObeyesdad']
lb = LabelEncoder()

# 라벨 인코딩할 열 목록
columns_to_encode = ['Gender','family_history_with_overweight','FAVC','CAEC','SMOKE','SCC','CALC','MTRANS']

# 데이터프레임 x의 열에 대해 라벨 인코딩 수행
for column in columns_to_encode:
    lb.fit(x[column])
    x[column] = lb.transform(x[column])

# 데이터프레임 test의 열에 대해 라벨 인코딩 수행
for column in columns_to_encode:
    lb.fit(test[column])
    test[column] = lb.transform(test[column])

# 데이터 분할
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=42)

# LightGBM 데이터셋 생성
train_data = lgb.Dataset(x_train, label=y_train)
test_data = lgb.Dataset(x_test, label=y_test)

# LightGBM 모델 파라미터 설정
import random
r = random.randint(1, 100)
random_state = r
params = {"objective": "multiclass",
               "metric": "multi_logloss",
               "verbosity": -1,
               "boosting_type": "gbdt",
               "random_state": random_state,
               "num_class": 7,
               "learning_rate" :  0.01386432121252535,
               'n_estimators': 500,         #에포
               'feature_pre_filter': False,
               'lambda_l1': 1.2149501037669967e-07,
               'lambda_l2': 0.9230890143196759,
               'num_leaves': 31,
               'feature_fraction': 0.5,
               'bagging_fraction': 0.5523862448863431,
               'bagging_freq': 4,
               'min_child_samples': 20}

# 모델 학습
num_round = 100
booster = lgb.train(params, train_data, num_round, valid_sets=[test_data])

# 모델 예측
y_pred = booster.predict(x_test)

# 확률값을 클래스로 변환
y_pred_class = [np.argmax(pred) for pred in y_pred]
accuracy = accuracy_score(y_test, y_pred_class)
y_submit = model.predict(test_csv)

print("Accuracy:", accuracy)
print("r",r)
y_submit = ohe.inverse_transform(y_submit)
y_submit = pd.DataFrame(y_submit)
sample_csv["대출등급"]=y_submit

sample_csv.to_csv(path + "대출87.csv", index=False)

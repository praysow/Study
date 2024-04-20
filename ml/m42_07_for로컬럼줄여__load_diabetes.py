from xgboost import XGBClassifier
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
import numpy as np

# 데이터 로드
x, y = load_diabetes(return_X_y=True)

# 데이터 분할
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=3)

# 데이터 스케일링
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# XGBClassifier 모델 정의
parameters = {
    'n_estimators': 4000,
    'learning_rate': 0.2,
    'max_depth': 3,
    'gamma': 4,
    'min_child_weight': 0.01,
    'subsample': 0.1,
    'colsample_bytree': 1,
    'colsample_bylevel': 1,
    'colsample_bynode': 1,
    'reg_alpha': 1,
    'reg_lambda': 1,
}
model = XGBClassifier()
model.set_params(early_stopping_rounds=10, **parameters)

# 이진 분류를 위한 레이블 변환
y_train_binary = np.where(y_train > np.median(y_train), 1, 0)
y_test_binary = np.where(y_test > np.median(y_train), 1, 0)

# 모델 훈련
model.fit(x_train, y_train_binary,
          eval_set=[(x_train, y_train_binary), (x_test, y_test_binary)],
          verbose=500,
          eval_metric='auc'
          )

# 테스트 세트에 대한 예측 및 평가
y_pred = model.predict(x_test)
f1 = f1_score(y_test_binary, y_pred)
print("f1_score:", f1)


#############
# print(model.feature_importances_)
# for문을 사용해서 피처가 약한놈부터 하나씩 제거
# 30,29,28,27....1
# 초기 평가
initial_f1 = f1_score(y_test_binary, model.predict(x_test))
print(f"Initial f1: {initial_f1}")

# Feature Importance를 이용한 피처 제거 및 평가
feature_importances = model.feature_importances_
sorted_indices = np.argsort(feature_importances)[::-1]

results = []

for i in range(len(sorted_indices)):
    # i개의 피처를 제거한 새로운 특징 배열 만들기
    reduced_x_train = np.delete(x_train, sorted_indices[:i+1], axis=1)
    reduced_x_test = np.delete(x_test, sorted_indices[:i+1], axis=1)
    
    # 만약 특징의 수가 0이 되면 반복문을 중단
    if reduced_x_train.shape[1] == 0:
        print("No features left to remove.")
        break
    
    # 모델 훈련
    model.fit(reduced_x_train, y_train_binary,
              eval_set=[(reduced_x_train, y_train_binary), (reduced_x_test, y_test_binary)],
              verbose=500,
              eval_metric='auc'
              )
    
    # 피처 제거 후 모델 평가
    f1 = f1_score(y_test_binary, model.predict(reduced_x_test))
    results.append((i+1, f1))

# 결과 출력
for result in results:
    print(f"After removing top {result[0]} features, f1: {result[1]}")

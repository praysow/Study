from xgboost import XGBClassifier,XGBRegressor
from sklearn.datasets import load_digits
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split,KFold, StratifiedKFold,GridSearchCV,RandomizedSearchCV,HalvingRandomSearchCV,HalvingGridSearchCV
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import numpy as np
from sklearn.metrics import accuracy_score,log_loss
#1. 데이터
x,y = load_digits(return_X_y=True)

x_train,x_test,y_train,y_test=train_test_split(x,y, train_size=0.8, random_state=3)

scaler=MinMaxScaler()
# scaler=StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

parameters = {
    'n_estimators' : 4000,
    'learning_rate' : 0.2,  #훈련량
    'max_depth' : 3,        #트리 노드의 깊이
    'gamma' : 4,
    'min_child_weight' : 0.01,
    'subsample' : 0.1,      # = dropout
    'colsample_bytree' : 1,
    'colsample_bylevel' : 1,
    'colsample_bynode' : 1,
    'reg_alpha' : 1,        # L1,L2 가중치 규제
    'reg_lambda' : 1,
    }
#2. 모델
model = XGBClassifier()
model.set_params(**parameters)
# #3. 훈련
model.fit(x_train,y_train,
          eval_set=[(x_train,y_train),(x_test,y_test)],
          verbose =1000,
          eval_metric='mlogloss'
          )
#4. 평가,예측
result = model.score(x_test,y_test)
print("최종점수:",result)
y_pred = model.predict(x_test)
acc = accuracy_score(y_test,y_pred)
print("acc:",acc)

############
#############
initial_loss = log_loss(y_test, model.predict_proba(x_test))
initial_accuracy = accuracy_score(y_test, model.predict(x_test))
print(f"Initial Log Loss: {initial_loss}, Initial Accuracy: {initial_accuracy}")

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
    model.fit(reduced_x_train, y_train,
              eval_set=[(reduced_x_train, y_train), (reduced_x_test, y_test)],
              verbose=1000,
              eval_metric='mlogloss'
              )
    
    # 피처 제거 후 모델 평가
    logloss = log_loss(y_test, model.predict_proba(reduced_x_test))
    accuracy = accuracy_score(y_test, model.predict(reduced_x_test))
    results.append((i+1, logloss, accuracy))

# 결과 출력
for result in results:
    print(f"After removing top {result[0]} features, Log Loss: {result[1]}, Accuracy: {result[2]}")
    
'''

'''
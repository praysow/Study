import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score,r2_score,log_loss

#1.데이터
x,y = load_breast_cancer(return_X_y=True)

x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.9,random_state=1)

scaler = MinMaxScaler()
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
model.set_params(early_stopping_rounds=10,**parameters)
#3. 훈련
model.fit(x_train,y_train,
          eval_set=[(x_train,y_train),(x_test,y_test)],
          verbose =10,
          eval_metric='logloss'
          )
#4. 평가,예측
result = model.score(x_test,y_test)
print("최종점수:",result)
y_pred = model.predict(x_test)
acc = accuracy_score(y_test,y_pred)
print("acc:",acc)

#############
# print(model.feature_importances_)
#for문을 사용해서 피처가 약한놈부터 하나씩 제거
#30,29,28,27....1
# 초기 평가
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
              verbose=10,
              eval_metric='logloss'
              )
    
    # 피처 제거 후 모델 평가
    logloss = log_loss(y_test, model.predict_proba(reduced_x_test))
    accuracy = accuracy_score(y_test, model.predict(reduced_x_test))
    results.append((i+1, logloss, accuracy))

# 결과 출력
for result in results:
    print(f"After removing top {result[0]} features, Log Loss: {result[1]}, Accuracy: {result[2]}")
'''
After removing top 1 features, Log Loss: 0.18701362058359247, Accuracy: 0.9649122807017544
After removing top 2 features, Log Loss: 0.20110638819463605, Accuracy: 0.9473684210526315
After removing top 3 features, Log Loss: 0.19254309281842558, Accuracy: 0.9473684210526315
After removing top 4 features, Log Loss: 0.22753407277135093, Accuracy: 0.9473684210526315
After removing top 5 features, Log Loss: 0.21483006155187642, Accuracy: 0.9473684210526315
After removing top 6 features, Log Loss: 0.21551944623890681, Accuracy: 0.9298245614035088
After removing top 7 features, Log Loss: 0.2270029514463689, Accuracy: 0.9298245614035088
After removing top 8 features, Log Loss: 0.2270029514463689, Accuracy: 0.9298245614035088
After removing top 9 features, Log Loss: 0.2270029514463689, Accuracy: 0.9298245614035088
After removing top 10 features, Log Loss: 0.2270029514463689, Accuracy: 0.9298245614035088
After removing top 11 features, Log Loss: 0.2270029514463689, Accuracy: 0.9298245614035088
After removing top 12 features, Log Loss: 0.2205289346868071, Accuracy: 0.9298245614035088
After removing top 13 features, Log Loss: 0.2205289346868071, Accuracy: 0.9298245614035088
After removing top 14 features, Log Loss: 0.2205289346868071, Accuracy: 0.9298245614035088
After removing top 15 features, Log Loss: 0.2205289346868071, Accuracy: 0.9298245614035088
After removing top 16 features, Log Loss: 0.21377915039664594, Accuracy: 0.9649122807017544
After removing top 17 features, Log Loss: 0.21377915039664594, Accuracy: 0.9649122807017544
After removing top 18 features, Log Loss: 0.2161117483153206, Accuracy: 0.9649122807017544
After removing top 19 features, Log Loss: 0.2161117483153206, Accuracy: 0.9649122807017544
After removing top 20 features, Log Loss: 0.2161117483153206, Accuracy: 0.9649122807017544
After removing top 21 features, Log Loss: 0.2161117483153206, Accuracy: 0.9649122807017544
After removing top 22 features, Log Loss: 0.2161117483153206, Accuracy: 0.9649122807017544
After removing top 23 features, Log Loss: 0.21829512377726476, Accuracy: 0.9649122807017544
After removing top 24 features, Log Loss: 0.21829512377726476, Accuracy: 0.9649122807017544
After removing top 25 features, Log Loss: 0.21829512377726476, Accuracy: 0.9649122807017544
After removing top 26 features, Log Loss: 0.21040551401223243, Accuracy: 0.9649122807017544
After removing top 27 features, Log Loss: 0.28257361966569877, Accuracy: 0.8771929824561403
After removing top 28 features, Log Loss: 0.28257361966569877, Accuracy: 0.8771929824561403
After removing top 29 features, Log Loss: 0.35208798534618047, Accuracy: 0.9122807017543859

'''
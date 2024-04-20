import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score,r2_score,log_loss

#1.데이터
x,y = load_breast_cancer(return_X_y=True)

x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.7,random_state=1)

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
# #4. 평가,예측
# result = model.score(x_test,y_test)
# print("최종점수:",result)
# y_pred = model.predict(x_test)
# acc = accuracy_score(y_test,y_pred)
# print("acc:",acc)

#############
print(model.feature_importances_)
#for문을 사용해서 피처가 약한놈부터 하나씩 제거
#30,29,28,27....1

thresholds = np.sort(model.feature_importances_)
print(thresholds)
print("----------------------------------------------------------------------------------------------------------------------------------")
from sklearn.feature_selection import SelectFromModel
for i in thresholds:
    selection = SelectFromModel(model,threshold=i, prefit=False)
    
    select_x_train = selection.transform(x_train)
    select_x_test = selection.transform(x_test)
    print("\t바뀐x_train",select_x_train.shape,"/","바뀐x_test",select_x_test.shape)
    
    select_model = XGBClassifier()
    select_model.set_params(early_stopping_rounds = 10,
                            **parameters,
                            eval_metric = 'logloss'
                            )
    
    select_model.fit(select_x_train, y_train,
                 eval_set=[(select_x_train, y_train), (select_x_test, y_test)],
                 verbose=0)

    select_y_pred = select_model.predict(select_x_test)
    score = accuracy_score(y_test,select_y_pred)
    
    print("trech=%.3f,n=%d,ACC:%.2f%%"%(i,select_x_train.shape[1],score*100))
    
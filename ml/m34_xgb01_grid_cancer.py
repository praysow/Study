from xgboost import XGBClassifier,XGBRegressor
from sklearn.datasets import load_breast_cancer
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split,KFold, StratifiedKFold,GridSearchCV,RandomizedSearchCV,HalvingRandomSearchCV,HalvingGridSearchCV
from sklearn.preprocessing import StandardScaler,MinMaxScaler

#1. 데이터
x,y = load_breast_cancer(return_X_y=True)

x_train,x_test,y_train,y_test=train_test_split(x,y, train_size=0.8, random_state=3)

scaler=MinMaxScaler()
scaler=StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits,shuffle=True,random_state=123)

# 'n_estimators' : [100,200,300] 디폴트 100/ 1~inf / 정수
# 'learning_rate' : [0.1,0.2,0.3] 디폴트 0.3/0~1/eta
# 'max_depth' : [None,2,3,4,5] 디폴트 6/0~inf/정수
# 'gamma' : [0,1,2,3,4] 디폴트 0 / 0~inf
# 'min_child_weight' : [0,0.001,0.01,0.1,0.5,1,5,10,100] 디폴트 1 /0~inf
# 'subsample' : [0,0.1,0.2,0.3,0.5,1] 디폴트 1 / 0~1
# 'colsample_bytree' : [0,0.1,0.2,0.3,1] 디폴트 1 / 0~1
# 'colsample_bylevel' : [0,0.1,0.2,0.3,1] 디폴트 1 / 0~1
# 'colsample_bynode' : [0,0.1,0.2,0.3,1] 디폴트 1 / 0~1
# 'reg_alpha' : [0,0.1,0.01,0.001,1,2,10] 디폴트 0 / 0~inf / L1 절대값 가중치 규제
# / alpha
# 'reg_lambda' : [0,0.1,0.01,0.001,1,2,10] 디폴트 1 / 0~inf / L1 제곱 가중치 가중치 규제
# / lambda

parameters = {
    'n_estimators' : [100,200,300] ,
    'learning_rate' : [0.1,0.2,0.3],
    'max_depth' : [None,2,3,4,5],
    'gamma' : [0,1,2,3,4],
    'min_child_weight' : [0,0.001,0.01,0.1,0.5,1,5,10,100],
    'subsample' : [0,0.1,0.2,0.3,0.5,1],
    'colsample_bytree' : [0,0.1,0.2,0.3,1],
    'colsample_bylevel' : [0,0.1,0.2,0.3,1],
    'colsample_bynode' : [0,0.1,0.2,0.3,1],
    'reg_alpha' : [0,0.1,0.01,0.001,1,2,10],
    'reg_lambda' : [0,0.1,0.01,0.001,1,2,10],
    'random_state' : [123]
    }
#2. 모델
xgb = XGBClassifier(random_state=123)
model = RandomizedSearchCV(xgb,parameters,cv=kfold,n_jobs=22)

#3. 훈련
model.fit(x_train,y_train)

#4. 평가,예측
print("최상의 매개변수 :", model.best_estimator_)
print("최상의 매개변수 :", model.best_params_)
print("최상의 점수 :", model.best_score_)

results = model.score(x_test,y_test)
y_pred = model.predict(x_test)
print("result",results)
'''
최상의 매개변수 : XGBClassifier(base_score=None, booster=None, callbacks=None,
              colsample_bylevel=0.2, colsample_bynode=0, colsample_bytree=0,
              device=None, early_stopping_rounds=None, enable_categorical=False,
              eval_metric=None, feature_types=None, gamma=1, grow_policy=None,
              importance_type=None, interaction_constraints=None,
              learning_rate=0.2, max_bin=None, max_cat_threshold=None,
              max_cat_to_onehot=None, max_delta_step=None, max_depth=None,
              max_leaves=None, min_child_weight=0.001, missing=nan,
              monotone_constraints=None, multi_strategy=None, n_estimators=200,
              n_jobs=None, num_parallel_tree=None, random_state=123, ...)
최상의 매개변수 : {'subsample': 1, 'reg_lambda': 0.001, 'reg_alpha': 0.001, 'random_state': 123, 'n_estimators': 200, 'min_child_weight': 0.001, 'max_depth': None, 'learning_rate': 0.2, 'gamma': 1, 'colsample_bytree': 0, 'colsample_bynode': 0, 'colsample_bylevel': 0.2}
최상의 점수 : 0.9714285714285715
result 0.9473684210526315
'''
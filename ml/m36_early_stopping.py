from xgboost import XGBClassifier,XGBRegressor
from sklearn.datasets import load_diabetes
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split,KFold, StratifiedKFold,GridSearchCV,RandomizedSearchCV,HalvingRandomSearchCV,HalvingGridSearchCV
from sklearn.preprocessing import StandardScaler,MinMaxScaler

#1. 데이터
x,y = load_diabetes(return_X_y=True)

x_train,x_test,y_train,y_test=train_test_split(x,y, train_size=0.8, random_state=3)

scaler=MinMaxScaler()
scaler=StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits,shuffle=True,random_state=123)

parameters = {
    'n_estimators' : 4000,
    'learning_rate' : 0.2,
    'max_depth' : 3,
    'gamma' : 4,
    'min_child_weight' : 0.01,
    'subsample' : 0.1,
    'colsample_bytree' : 1,
    'colsample_bylevel' : 1,
    'colsample_bynode' : 1,
    'reg_alpha' : 1,
    'reg_lambda' : 1,
    }
#2. 모델
model = XGBRegressor(**parameters)
# model = RandomizedSearchCV(xgb,parameters,cv=kfold,n_jobs=22)

#3. 훈련
model.fit(x_train,y_train,
          eval_set=[(x_test,y_test)],
          early_stopping_rounds=5,
          verbose =10
          )

# model.set_params(
#     # **parameters,
#     # early_stopping_rounds=5,
#     # learning_rate=0.01,
#     # n_estimators = 4000,
#     # max_depth = 8,
#     # random_state=349,
#     # reg_alpha = 101,
#     # reg_lambda =101
#     )

result = model.score(x_test,y_test)
print("최종점수3:",result)
print("사용파라미터",model.get_params())
'''
[98]    validation_0-rmse:64.96701
[99]    validation_0-rmse:64.96810
최종점수3: 0.22048206817710214
xgboost의 디폴트는 rmse이다
'''
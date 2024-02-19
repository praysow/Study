from xgboost import XGBClassifier,XGBRegressor
from sklearn.datasets import load_diabetes,load_breast_cancer
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
# model = XGBRegressor(**parameters)
xg = XGBClassifier(**parameters)
model = RandomizedSearchCV(xg,parameters, cv = kfold,verbose=1,refit=True,n_jobs=-1)       #n_jobs gpu아니고 cpu

#3. 훈련
model.fit(x_train,y_train,
          eval_set=[(x_test,y_test)],
        #   eval_metric = 'rmse', # 디폴트
          eval_metric = 'mae',  #rmsle, mape, mphe... 등 (회귀)
        #   eval_metric = 'error',# 이진분류
        #   eval_metric = 'merror',# 다중분류
        #   eval_metric = 'logloss', #이진분류 디폴트
        #   eval_metric = 'mlogloss', #다중분류 디폴트
        #   eval_metric = 'auc',    #이중분류 디폴트
          early_stopping_rounds=10,
          verbose =1
          )

from sklearn.metrics import r2_score,accuracy_score,f1_score,roc_auc_score,mean_absolute_error
result = model.score(x_test,y_test)
y_pred = model.predict(x_test)
print("최종점수:",result)
print("사용파라미터",model.get_params())
# r2=r2_score(y_test,y_pred)
# accuracy = accuracy_score(y_test,y_pred)
# f1 = f1_score(y_test,y_pred)
# auc = roc_auc_score(y_test,y_pred)
print('r2:',r2_score(y_test,y_pred))
print('acc',accuracy_score(y_test,y_pred))
print('f1',f1_score(y_test,y_pred))
print('auc',roc_auc_score(y_test,y_pred))
# hist = model.evals_result()
# print(hist)
'''
[98]    validation_0-rmse:64.96701
[99]    validation_0-rmse:64.96810
최종점수3: 0.22048206817710214
xgboost의 디폴트는 rmse이다
'''
best_model = model.best_estimator_
print("Best Model:", best_model)
print("Best Score:", model.best_score_)

# 최적 모델에서 evals_result() 호출
evals_result = best_model.evals_result()
print("Eval Result:", evals_result)

# 그래프 그리기
import matplotlib.pyplot as plt
plt.plot(evals_result['validation_0']['mae'], label='train',color='red')
plt.plot(evals_result['validation_0']['mae'], label='test',color='blue')
plt.xlabel('epochs')
plt.ylabel('mae')
plt.title('XGBoost MAE')
plt.legend()
plt.show()


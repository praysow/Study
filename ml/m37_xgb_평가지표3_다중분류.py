from xgboost import XGBClassifier,XGBRegressor
from sklearn.datasets import load_digits,load_breast_cancer
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split,KFold, StratifiedKFold,GridSearchCV,RandomizedSearchCV,HalvingRandomSearchCV,HalvingGridSearchCV
from sklearn.preprocessing import StandardScaler,MinMaxScaler

#1. 데이터
x,y = load_digits(return_X_y=True)

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
# model = XGBRegressor(**parameters)
model = XGBClassifier(**parameters)

#3. 훈련
model.fit(x_train,y_train,
          eval_set=[(x_test,y_test)],
        #   eval_metric = 'rmse', # 디폴트
        #   eval_metric = 'mae',  #rmsle, mape, mphe... 등 (회귀)
        #   eval_metric = 'error',# 이진분류
        #   eval_metric = 'merror',# 다중분류
        #   eval_metric = 'logloss', #이진분류 디폴트
        #   eval_metric = 'mlogloss', #다중분류 디폴트
        #   eval_metric = 'auc',    #이중분류 디폴트
          early_stopping_rounds=5,
          verbose =10
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

'''
RMSE (Root Mean Squared Error):
회귀 문제에서 사용되는 평가 지표로, 예측 값과 실제 값 간의 차이의 제곱의 평균을 측정한 뒤 이를 제곱근으로 변환한 값입니다. 예측 값과 실제 값 간의 거리를 나타내는 지표로, 값이 작을수록 모델의 예측이 정확합니다.

MAE (Mean Absolute Error):
회귀 문제에서 사용되는 평가 지표로, 예측 값과 실제 값 간의 차이의 절대값의 평균을 측정합니다. 예측 값과 실제 값 간의 평균적인 절대적인 차이를 나타내는 지표로, 값이 작을수록 모델의 예측이 정확합니다.

Error:
분류 문제에서 잘못 분류된 데이터의 비율을 나타내는 평가 지표입니다. 이 값이 작을수록 모델의 분류가 정확합니다.

MError (Multiclass Error):
다중 클래스 분류 문제에서 잘못 분류된 데이터의 비율을 나타내는 평가 지표입니다. 이 값이 작을수록 모델의 분류가 정확합니다.

Logloss (Logarithmic Loss):
이진 분류 및 다중 클래스 분류 문제에서 사용되는 평가 지표로, 모델이 예측한 확률 분포와 실제 레이블 간의 차이를 측정합니다. 값이 작을수록 모델의 예측이 정확합니다.

mLogloss (Multiclass Logarithmic Loss):
다중 클래스 분류 문제에서 사용되는 평가 지표로, Logloss와 유사하지만 다중 클래스에 대해 확장된 지표입니다.

AUC (Area Under the Curve):
 이진 분류 문제에서 사용되는 평가 지표로, ROC 곡선 아래 영역을 나타냅니다. ROC 곡선은 True Positive Rate와 False Positive Rate 간의 관계를 나타내며, AUC가 높을수록 모델의 성능이 좋습니다.
'''
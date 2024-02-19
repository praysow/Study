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
# print('f1',f1_score(y_test,y_pred))
# print('auc',roc_auc_score(y_test,y_pred))

#########################################
import pickle
import joblib
path = 'c:/_data/_save/_pickle_test/'
pickle.dump(model, open(path + 'm39_pickle1_save.dat1','wb'))
joblib.dump(model, open(path + 'm39_pickle1_save.dat2','wb'))

'''
r2: 0.6374505025890953
acc 0.8583333333333333
'''

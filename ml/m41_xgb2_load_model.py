from xgboost import XGBClassifier,XGBRegressor
from sklearn.datasets import load_digits,load_breast_cancer
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split,KFold, StratifiedKFold,GridSearchCV,RandomizedSearchCV,HalvingRandomSearchCV,HalvingGridSearchCV
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import pickle as pk
import joblib
#1. 데이터
x,y = load_digits(return_X_y=True)

x_train,x_test,y_train,y_test=train_test_split(x,y, train_size=0.8, random_state=3)

scaler=MinMaxScaler()
scaler=StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

path = 'c:/_data/_save/_joblib_test/'
model = XGBClassifier()
model.load_model(path+'m41_xgb1_save_model.dat')


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

'''
r2: 0.6374505025890953
acc 0.8583333333333333
'''

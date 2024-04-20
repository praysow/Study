import numpy as np
import pandas as pd
from keras.models import Sequential, load_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler, Normalizer, RobustScaler
from sklearn.metrics import accuracy_score, f1_score ,log_loss
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from xgboost import XGBClassifier
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split,KFold, StratifiedKFold,GridSearchCV,RandomizedSearchCV,HalvingRandomSearchCV,HalvingGridSearchCV

path= "c:/_data/kaggle/비만/"
train=pd.read_csv(path+"train.csv",index_col=0)
test=pd.read_csv(path+"test.csv",index_col=0)
sample=pd.read_csv(path+"sample_submission.csv")
x= train.drop(['NObeyesdad'],axis=1)
y= train['NObeyesdad']
# print(train.shape,test.shape)   #(20758, 17) (13840, 16)    NObeyesdad
# print(x.shape,y.shape)  #(20758, 16) (20758,)

lb = LabelEncoder()
y = lb.fit_transform(train['NObeyesdad'])
# 라벨 인코딩할 열 목록
columns_to_encode = ['Gender','family_history_with_overweight','FAVC','CAEC','SMOKE','SCC','CALC','MTRANS']

# 데이터프레임 x의 열에 대해 라벨 인코딩 수행
for column in columns_to_encode:
    lb.fit(x[column])
    x[column] = lb.transform(x[column])

# 데이터프레임 test_csv의 열에 대해 라벨 인코딩 수행
for column in columns_to_encode:
    lb.fit(test[column])
    test[column] = lb.transform(test[column])
x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.9,random_state=1)
from bayes_opt import BayesianOptimization
import time
bayesian_params = {
    'learning_rate' : (0.001,1),
    'max_depth' : (3,10),
    'num_leaves' : (24,40),
    'min_child_samples' : (10,200),
    'min_child_weight': (1,50),
    'subsample' : (0.5,1),
    'colsample_bytree' : (0.5,1),
    'max_bin' : (9,500),
    'reg_lambda' : (-0.001,10),
    'reg_alpha': (0.01,50)
}

def xgb_hamsu(learning_rate,max_depth,num_leaves,min_child_samples,min_child_weight,subsample,colsample_bytree,max_bin,reg_lambda,reg_alpha):
    params = {
    'n_estimators' : 100,
    'learning_rate' : learning_rate,
    'max_depth' : int(round(max_depth)),
    'num_leaves' : int(round(num_leaves)),
    'min_child_samples' : int(round(min_child_samples)),
    'min_child_weight': int(round(min_child_weight)),
    'subsample' : max(min(subsample,1),0),
    'colsample_bytree' : colsample_bytree,
    'max_bin' : max(int(round(max_bin)),10),
    'reg_lambda' : max(reg_lambda,0),
    'reg_alpha': reg_alpha
    }
    model = XGBClassifier(**params,n_jobs = -1)
    model.fit(x_train,y_train,
              eval_set=[(x_train,y_train),(x_test,y_test)],
              eval_metric='mlogloss',
              verbose = 0,
              early_stopping_rounds= 50
              )
    y_pred = model.predict(x_test)
    result = accuracy_score(y_test,y_pred)
    return result
s_t = time.time()
bay = BayesianOptimization(
    f= xgb_hamsu,
    pbounds=bayesian_params,
    random_state=777   
    )
n_iter = 100
bay.maximize(init_points=5,n_iter=n_iter)
e_t = time.time()
print(bay.max)
print("시간:",round((e_t-s_t),2),"초")
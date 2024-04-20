import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import Perceptron,LogisticRegression,SGDClassifier
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.utils import all_estimators
import warnings
from sklearn.metrics import accuracy_score
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV,HalvingGridSearchCV,HalvingRandomSearchCV
warnings.filterwarnings('ignore')
import time
#1. 데이터

path= "c:\_data\kaggle\\bike\\"
train_csv = pd.read_csv(path+"train.csv",index_col=0)
test_csv = pd.read_csv(path+"test.csv",index_col=0)
sampleSubmission_csv = pd.read_csv(path+"sampleSubmission.csv")

x= train_csv.drop(['count','casual','registered'], axis=1)
y= train_csv['count']

x_train,x_test,y_train,y_test=train_test_split(x,y, train_size=0.8, random_state=3)
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
# 데이터 스케일링
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# XGBRegressor 모델 정의
parameters = {
    'n_estimators': 4000,
    'learning_rate': 0.2,
    'max_depth': 3,
    'gamma': 4,
    'min_child_weight': 0.01,
    'subsample': 0.1,
    'colsample_bytree': 1,
    'colsample_bylevel': 1,
    'colsample_bynode': 1,
    'reg_alpha': 1,
    'reg_lambda': 1,
}
model = XGBRegressor()
model.set_params(early_stopping_rounds=10, **parameters)

# 모델 훈련
model.fit(x_train, y_train,
          eval_set=[(x_train, y_train), (x_test, y_test)],
          verbose=500,
          eval_metric='auc'
          )

# 테스트 세트에 대한 예측 및 평가
y_pred = model.predict(x_test)
median_y_train = np.median(y_train)
f1 = f1_score(y_test, np.where(y_pred > median_y_train, 1, 0), average=None)
print("f1_score:", f1)


# Feature Importance를 이용한 피처 제거 및 평가
thresholds = np.sort(model.feature_importances_)
print(thresholds)
print("----------------------------------------------------------------------------------------------------------------------------------")
from sklearn.feature_selection import SelectFromModel
for i in thresholds:
    selection = SelectFromModel(model,threshold=i, prefit=False)
    
    select_x_train = selection.transform(x_train)
    select_x_test = selection.transform(x_test)
    print("\t바뀐x_train",select_x_train.shape,"/","바뀐x_test",select_x_test.shape)
    
    select_model = XGBRegressor()
    select_model.set_params(early_stopping_rounds = 10,
                            **parameters,
                            eval_metric = 'auc'
                            )
    
    select_model.fit(select_x_train, y_train,
                 eval_set=[(select_x_train, y_train), (select_x_test, y_test)],
                 verbose=0)

    select_y_pred = select_model.predict(select_x_test)
    score = f1_score(y_test,select_y_pred)
    
    print("trech=%.3f,n=%d,ACC:%.2f%%"%(i,select_x_train.shape[1],score*100))
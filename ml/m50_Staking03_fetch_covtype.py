import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import warnings
from sklearn.metrics import accuracy_score,log_loss
warnings.filterwarnings('ignore')
import time
from sklearn.datasets import fetch_covtype

#1.데이터
datasets= fetch_covtype()
x= datasets.data
y= datasets.target

y -= 1

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.86,
                                                    random_state=5,        #346
                                                    # stratify=y_ohe1            
                                                    )
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from xgboost import XGBClassifier
scaler=MinMaxScaler()
scaler=StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
from sklearn.ensemble import RandomForestClassifier,StackingClassifier
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
xgb = XGBClassifier()
rf = RandomForestClassifier()
log = LogisticRegression()
cat = CatBoostClassifier(verbose=0)
model = StackingClassifier(
    estimators=[('xgb',xgb),('rf',rf),('log',log)],final_estimator=cat,
    n_jobs=-1,
    cv=5,
    )
# 모델 훈련
model.fit(x_train,y_train)

# 결과 예측
result = model.score(x_test,y_test)
y_pred = model.predict(x_test)
acc = accuracy_score(y_test,y_pred)
print(result)
print("acc",acc)

'''
0.9634383221460008
acc 0.9634383221460008
'''

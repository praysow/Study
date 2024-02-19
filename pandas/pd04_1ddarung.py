import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.utils import all_estimators
import warnings
from sklearn.metrics import accuracy_score
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV,HalvingGridSearchCV,HalvingRandomSearchCV
warnings.filterwarnings('ignore')
import time
from xgboost import XGBRegressor
#1. 데이터

path= "c:\_data\dacon\ddarung\\"

train_csv = pd.read_csv(path+"train.csv",index_col=0)
test_csv= pd.read_csv(path+"test.csv",index_col=0)
submission_csv= pd.read_csv(path+"submission.csv")


print(train_csv.columns)
x= train_csv.drop(['count'],axis=1)
y= train_csv['count']
z= train_csv['hour_bef_pm2.5']

# print(pd.value_counts(z))
# print(np.unique(z,return_counts=True))
# 'hour'
# hour_bef_temperature: 이전 시간에 기록된 온도18.8    17
# 19.4    17
# 14.0    16
# 18.0    16
# 16.6    15
#         ..
# 3.2      1
# 5.3      1
# 7.6      1
# 6.3      1
# 29.2     1
# hour_bef_precipitation: 이전 시간에 기록된 강수량([ 0.,  1., nan])
# hour_bef_windspeed: 이전 시간에 기록된 풍속
# (array([0. , 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. , 1.1, 1.2,
#        1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2. , 2.1, 2.2, 2.3, 2.4, 2.5,
#        2.6, 2.7, 2.8, 2.9, 3. , 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8,
#        3.9, 4. , 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5. , 5.1,
#        5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 6. , 6.2, 6.3, 6.4, 6.5,
#        6.7, 7. , 7.1, 7.3, 7.4, 7.5, 7.7, 8. , nan])
# hour_bef_humidity: 이전 시간에 기록된 습도
# [ 7.,  8.,  9., 10., 11., 12., 13., 14., 15., 16., 17., 18., 19.,
#        20., 21., 22., 23., 24., 25., 26., 27., 28., 29., 30., 31., 32.,
#        33., 34., 35., 36., 37., 38., 39., 40., 41., 42., 43., 44., 45.,
#        46., 47., 48., 49., 50., 51., 52., 53., 54., 55., 56., 57., 58.,
#        59., 60., 61., 62., 63., 64., 65., 66., 67., 68., 69., 70., 71.,
#        72., 73., 74., 75., 76., 77., 78., 79., 80., 81., 82., 83., 84.,
#        85., 86., 87., 88., 89., 90., 91., 92., 93., 94., 95., 98., 99.,
#        nan])
# hour_bef_visibility: 이전 시간에 기록된 가시성
# hour_bef_ozone: 이전 시간에 기록된 오존 농도
# hour_bef_pm10: 이전 시간에 기록된 미세먼지(PM10) 농도
# hour_bef_pm2.5: 이전 시간에 기록된 초미세먼지(PM2.5) 농도
# (array([ 8.,  9., 10., 11., 12., 13., 14., 15., 16., 17., 18., 19., 20.,
#        21., 22., 23., 24., 25., 26., 27., 28., 29., 30., 31., 32., 33.,
#        34., 35., 36., 37., 38., 39., 40., 41., 42., 43., 44., 45., 46.,
#        47., 48., 49., 50., 51., 52., 53., 54., 55., 56., 57., 58., 59.,
#        60., 61., 62., 63., 64., 65., 66., 67., 68., 69., 70., 71., 72.,
#        73., 74., 75., 77., 78., 79., 81., 82., 83., 84., 85., 86., 89.,
#        90., nan]),
# 'count'
# hour를 제외하고는 전부 이상치 존재

x_train,x_test,y_train,y_test=train_test_split(x,y, train_size=0.8, random_state=6)

# def outliers(data_out):
#     quartile_1,q2,quartile_3 = np.percentile(data_out,[25,50,75])
#     print("1사분위 :", quartile_1)
#     print("q2",q2)
#     print("3사분위 :", quartile_3)
#     iqr = quartile_3 - quartile_1
#     print("iqr:",iqr)
#     lower_bound = quartile_1 - (iqr*1.5)    # *1.5 이걸만든 프로그래머가 정한수치이고 직접 조정해도 된다(범위를 조금 늘리기 위해서 해주는것)
#     upper_bound = quartile_3 + (iqr*1.5)
#     return np.where((data_out>upper_bound) |    # |는 또는이라는 뜻이다
#                     (data_out<lower_bound))



# outliers_loc = outliers(z)
# print("이상치의 위치 :",outliers_loc)

train_csv = train_csv.dropna()
# train_csv=train_csv.fillna(train_csv.mean())                        
# train_csv=train_csv.fillna(0)
# test_csv=test_csv.fillna(test_csv.mean())
test_csv=test_csv.fillna(0)
#대체적으로 데이터간의 편차가 크기 때문에 dropna 와 fillna(0)으로 데이터 처리

#2.모델훈련
model = XGBRegressor(
    booster='gbtree',
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    subsample=1.0,
    colsample_bytree=1.0,
    gamma=0,
    reg_alpha=0,
    reg_lambda=1,
    scale_pos_weight=1,
    eval_metric='mlogloss',
    early_stopping_rounds=None,
    verbosity=1,
    random_state=None,
    n_jobs=None
)
model.fit(x_train, y_train)
#3.평가, 예측
result = model.score(x_test,y_test)
y_predict = model.predict(x_test)
from sklearn.metrics import r2_score
r2=r2_score(y_test,y_predict)
y_submit = model.predict(test_csv)
submission_csv['count'] = y_submit
submission_csv.to_csv(path + "submission_21.csv", index=False)
# print(submission_csv)
print("R2 score",r2)
print("R2 score:", result)
'''
R2 score 0.7865920770848444
R2 score: 0.7865920770848444

로스 : 2694.685302734375
10/10 [==============================] - 0s 0s/step
R2 score 0.5969105620875779
'''
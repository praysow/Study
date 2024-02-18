from sklearn.metrics import r2_score, mean_squared_error, mean_squared_log_error
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

#1. 데이터

path= "c:\_data\kaggle\\bike\\"
train_csv = pd.read_csv(path+"train.csv",index_col=0)
test_csv = pd.read_csv(path+"test.csv",index_col=0)
sampleSubmission_csv = pd.read_csv(path+"sampleSubmission.csv")

# print(train_csv)
# print(test_csv)
# print(submission_csv)

# print("train",train_csv.shape)      #(10886, 11)
# print("test",test_csv.shape)       #(6493, 8)
# print("sub",sampleSubmission_csv.shape) #(6493, 2)

#train_csv=train_csv.dropna()
# train_csv=train_csv.fillna(train_csv.mean())
# train_csv=train_csv.fillna(0)
# test_csv=test_csv.fillna(test_csv.mean())
#test_csv=test_csv.fillna(0)

x= train_csv.drop(['count','casual','registered'], axis=1)
y= train_csv['count']
# z= train_csv['windspeed']
# print(pd.value_counts(z))
# # print(np.unique(z,return_counts=True))
# # print(train_csv.columns)Index(['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp','humidity', 'windspeed', 'casual', 'registered', 'count'],dtype='object')
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
x_train,x_test,y_train,y_test=train_test_split(x,y, train_size=0.8, random_state=6)

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
sampleSubmission_csv['count'] = y_submit
sampleSubmission_csv.to_csv(path + "submission_21.csv", index=False)
print("R2 score",r2)
print("acc :", result)
'''
R2 score 0.3211879989617036
acc : 0.3211879989617036
'''

#결측치는 없고 별다른 이상치는 보이지 않아서 데이터 전처리를 거의 하지 않음
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import r2_score

# 데이터 로드
path = "c:\_data\dacon\ddarung\\"
train_csv = pd.read_csv(path+"train.csv", index_col=0)
test_csv = pd.read_csv(path+"test.csv", index_col=0)
submission_csv = pd.read_csv(path+"submission.csv")

# 결측치 처리
train_csv = train_csv.fillna(train_csv.mean())
test_csv = test_csv.fillna(test_csv.mean())

# 특성과 타겟 분리
x = train_csv.drop(['count'], axis=1)
y = train_csv['count']

x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.9,random_state=1)
from sklearn.preprocessing import MinMaxScaler
sclaer = MinMaxScaler().fit(x_train)
x_train = sclaer.transform(x_train)
x_test = sclaer.transform(x_test)
from xgboost import XGBRegressor
# model 
from sklearn.ensemble import VotingRegressor, RandomForestRegressor
model = VotingRegressor([
    # ('LR',LogisticRegression()),
    ('RF',RandomForestRegressor()),
    ('XGB',XGBRegressor()),
    ])

# fit & pred
model.fit(x_train,y_train,)

result = model.score(x_test,y_test)
print("Score: ",result)

pred = model.predict(x_test)
acc = r2_score(y_test,pred)
print("ACC: ",acc)
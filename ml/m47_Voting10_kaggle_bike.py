import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import warnings
from sklearn.metrics import r2_score
warnings.filterwarnings('ignore')
import time
#1. 데이터

path= "c:\_data\kaggle\\bike\\"
train_csv = pd.read_csv(path+"train.csv",index_col=0)
test_csv = pd.read_csv(path+"test.csv",index_col=0)
sampleSubmission_csv = pd.read_csv(path+"sampleSubmission.csv")

x= train_csv.drop(['count','casual','registered'], axis=1)
y= train_csv['count']

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
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
#1. 데이터

path= "c:\_data\dacon\ddarung\\"

train_csv = pd.read_csv(path+"train.csv",index_col=0)
test_csv= pd.read_csv(path+"test.csv",index_col=0)
submission_csv= pd.read_csv(path+"submission.csv")

# train_csv=train_csv.fillna(train_csv.mean())                        
# train_csv=train_csv.fillna(0)
# test_csv=test_csv.fillna(test_csv.mean())                         #test는 dropna를 하면 안되고 결측치를 변경해줘야한다
# test_csv=test_csv.fillna(0)
print(train_csv.columns)#['hour', 'hour_bef_temperature', 'hour_bef_precipitation','hour_bef_windspeed', 'hour_bef_humidity', 'hour_bef_visibility','hour_bef_ozone', 'hour_bef_pm10', 'hour_bef_pm2.5', 'count']
x= train_csv.drop(['count'],axis=1)
y= train_csv['count']
z= train_csv['hour_bef_windspeed']
#hour:이상치의 위치 : (array([], dtype=int64),)
# 'hour'
# 'hour_bef_temperature
# '1사분위 : nan
# q2 nan
# 3사분위 : nan
# iqr: nan
# 이상치의 위치 : (array([], dtype=int64),)
# 'hour_bef_precipitation'
# 1사분위 : nan
# q2 nan
# 3사분위 : nan
# iqr: nan
# 이상치의 위치 : (array([], dtype=int64),)
# 'hour_bef_windspeed',
# 'hour_bef_humidity',
# 'hour_bef_visibility',
# 'hour_bef_ozone',
# 'hour_bef_pm10',
# 'hour_bef_pm2.5',
# 'count'

# x_train,x_test,y_train,y_test=train_test_split(x,y, train_size=0.8, random_state=6)

def outliers(data_out):
    quartile_1,q2,quartile_3 = np.percentile(data_out,[25,50,75])
    print("1사분위 :", quartile_1)
    print("q2",q2)
    print("3사분위 :", quartile_3)
    iqr = quartile_3 - quartile_1
    print("iqr:",iqr)
    lower_bound = quartile_1 - (iqr*1.5)    # *1.5 이걸만든 프로그래머가 정한수치이고 직접 조정해도 된다(범위를 조금 늘리기 위해서 해주는것)
    upper_bound = quartile_3 + (iqr*1.5)
    return np.where((data_out>upper_bound) |    # |는 또는이라는 뜻이다
                    (data_out<lower_bound))



outliers_loc = outliers(z)
print("이상치의 위치 :",outliers_loc)

# import matplotlib.pyplot as plt
# plt.boxplot(z)
# plt.show()
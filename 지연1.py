import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import lightgbm as lgb
import optuna

data = 'c:/_data/project/mini/지연.py'

x = data.drop('최대지연시간')
y = data['최대지연시간']

lb = LabelEncoder()

# 라벨 인코딩할 열 목록
columns_to_encode = ['지연일자','노선','지연시간대']

# 데이터프레임 x의 열에 대해 라벨 인코딩 수행
for column in columns_to_encode:
    lb.fit(x[column])
    x[column] = lb.transform(x[column])

# # 데이터프레임 test_csv의 열에 대해 라벨 인코딩 수행
# for column in columns_to_encode:
#     lb.fit(test[column])
#     test[column] = lb.transform(test[column])

print(x)
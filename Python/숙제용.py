from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
#1. 데이터
path= "c:\_data\dacon\wine\\"
train = pd.read_csv(path+"train.csv",index_col=0)
test = pd.read_csv(path+"test.csv",index_col=0)
sample = pd.read_csv(path+"sample_Submission.csv")
x= train.drop(['quality'], axis=1)
y= train['quality']

# print(x.describe())
#        fixed acidity  volatile acidity  citric acid  residual sugar    chlorides  free sulfur dioxide  total sulfur dioxide      density           pH    sulphates      alcohol
# count    5497.000000       5497.000000  5497.000000     5497.000000  5497.000000          5497.000000           5497.000000  5497.000000  5497.000000  5497.000000  5497.000000
# mean        7.210115          0.338163     0.318543        5.438075     0.055808            30.417682            115.566491     0.994673     3.219502     0.530524    10.504918
# std         1.287579          0.163224     0.145104        4.756676     0.034653            17.673881             56.288223     0.003014     0.160713     0.149396     1.194524
# min         3.800000          0.080000     0.000000        0.600000     0.009000             1.000000              6.000000     0.987110     2.740000     0.220000     8.000000
# 25%         6.400000          0.230000     0.250000        1.800000     0.038000            17.000000             78.000000     0.992300     3.110000     0.430000     9.500000
# 50%         7.000000          0.290000     0.310000        3.000000     0.047000            29.000000            118.000000     0.994800     3.210000     0.510000    10.300000
# 75%         7.700000          0.400000     0.390000        8.100000     0.064000            41.000000            155.000000     0.996930     3.320000     0.600000    11.300000
# max        15.900000          1.580000     1.660000       65.800000     0.610000           289.000000            440.000000     1.038980     4.010000     2.000000    14.900000

lb=LabelEncoder()
lb.fit(x['type'])
x['type'] =lb.transform(x['type'])
test['type'] =lb.transform(test['type'])

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, random_state=367, stratify=y,shuffle=True)
from sklearn.preprocessing import StandardScaler
scaler =StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
test = scaler.transform(test)
# import random
# r = random.randint(1, 100)
# random_state = r
# # 모델 생성 및 학습
# lgbm_params = {
#             #    "objective": "multiclass",
#             #    "metric": "multi_logloss",
#             #    "verbosity": -1,
#             #    "boosting_type": "gbdt",
#             #    "random_state": random_state,
#             #    "num_class": 7,
#             #    "learning_rate": 0.01386432121252535,
#             #    "n_estimators": 500,
#             #    "feature_pre_filter": False,
#             #    "lambda_l1": 1.2149501037669967e-07,
#             #    "lambda_l2": 0.9230890143196759,
#             #    "num_leaves": 31,
#             #    "feature_fraction": 0.5,
#             #    "bagging_fraction": 0.5523862448863431,
#             #    "bagging_freq": 4,
#             #    "min_child_samples": 20,
#             #    "max_depth":12,
#             #    "min_samples_leaf":10,
#             #    'n_jobs': -1
#                }


# # model = LGBMClassifier(**lgbm_params,device='gpu')
# model = RandomForestClassifier()
# model.fit(x_train, y_train)

# # 모델 저장
# # booster = model.booster_
# # model.booster_.save_model("c:/_data/_save/와인.h5")

# # 테스트 데이터 예측 및 저장
# y_pred = model.predict(x_test)
# y_submit = model.predict(test)
# # sample['quality'] = y_submit
# # sample.to_csv(path + "와인.csv", index=False)

# # 정확도 평가
# accuracy = accuracy_score(y_test, y_pred)
# print("Accuracy:", accuracy)
# print("r",r)
from sklearn.tree import DecisionTreeClassifier,plot_tree
import matplotlib.pyplot as plt
model = DecisionTreeClassifier()
model.fit(x, y)

# 결정 트리 시각화
plt.figure(figsize=(15, 10))
plot_tree(model)
plt.show()
import numpy as np
import pandas as pd
from keras.models import Sequential, load_model
from keras.layers import Dense,Dropout,BatchNormalization, AveragePooling1D, Flatten, Conv2D, LSTM, Bidirectional,Conv1D,MaxPooling1D
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler, Normalizer, RobustScaler
from sklearn.metrics import accuracy_score, f1_score
from lightgbm import LGBMClassifier

path= "c:/_data/kaggle/비만/"
train=pd.read_csv(path+"train.csv",index_col=0)
test=pd.read_csv(path+"test.csv",index_col=0)
sample=pd.read_csv(path+"sample_submission.csv")
x= train.drop(['NObeyesdad'],axis=1)
y= train['NObeyesdad']
# print(train.shape,test.shape)   #(20758, 17) (13840, 16)    NObeyesdad
# print(x.shape,y.shape)  #(20758, 16) (20758,)

lb = LabelEncoder()

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
    
# print(x['Gender'])
# print(test['CALC'])
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.90,random_state=3,stratify=y)

# print(x_train.shape,y_train.shape)  #(18682, 16) (18682,)
# print(x_test.shape,y_test.shape)    #(2076, 16) (2076,)
import random

r = random.randint(1, 100)
random_state = r
lgbm_params = {"objective": "multiclass",
               "metric": "multi_logloss",
               "verbosity": -1,
               "boosting_type": "gbdt",
               "random_state": random_state,
               "num_class": 7,
               "learning_rate" :  0.01386432121252535,
               'n_estimators': 500,         #에포
               'feature_pre_filter': False,
               'lambda_l1': 1.2149501037669967e-07,
               'lambda_l2': 0.9230890143196759,
               'num_leaves': 31,
               'feature_fraction': 0.5,
               'bagging_fraction': 0.5523862448863431,
               'bagging_freq': 4,
               'min_child_samples': 20}

model = LGBMClassifier(**lgbm_params)

# 모델 학습
model.fit(x_train, y_train)
model.booster_.save_model("c:\_data\_save\비만5.h5")
# 테스트 데이터에 대한 예측
y_pred = model.predict(x_test)
y_submit = model.predict(test)
sample['NObeyesdad']=y_submit

sample.to_csv(path + "비만5.csv", index=False)
# 정확도 평가
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("r",r)
'''
Accuracy: 0.9142581888246628        903등
Accuracy: 0.9185934489402697    4번
'''

# 이 코드에서 사용된 LightGBM 모델의 파라미터들에 대해 간단히 설명하겠습니다.

# objective: 최적화 목표를 설정합니다. 여기서는 다중 클래스 분류를 수행하므로 "multiclass"로 설정됩니다.
# metric: 모델의 성능을 측정하는 메트릭을 지정합니다. 이 경우 다중 클래스 분류의 경우 다중 로그 손실인 "multi_logloss"가 사용됩니다.
# verbosity: 학습 중에 출력되는 메시지의 상세도를 조절합니다. -1로 설정하면 아무 메시지도 출력되지 않습니다.
# boosting_type: 부스팅 방법을 선택합니다. 여기서는 기본값인 Gradient Boosting Decision Tree인 "gbdt"가 사용됩니다.
# random_state: 모델 학습 시 무작위성을 제어하기 위한 시드(seed) 값입니다. 모델의 재현성을 위해 설정됩니다.
# num_class: 다중 클래스 분류에서 클래스의 개수를 지정합니다. 이 경우에는 7개의 클래스가 있으므로 7로 설정됩니다.
# learning_rate: 학습률(learning rate)로, 각 트리가 기존 트리의 오차를 얼마나 강하게 보정할지를 결정합니다.
# n_estimators: 사용할 트리의 개수를 지정합니다. 부스팅 반복 횟수입니다.
# feature_pre_filter: 특징량 사전 필터링 여부를 결정합니다. 여기서는 False로 설정되어 있으므로 사용되지 않습니다.
# lambda_l1, lambda_l2: L1 및 L2 정규화 파라미터입니다. 모델의 과적합을 방지하기 위해 사용됩니다.
# num_leaves: 각 트리의 최대 잎 노드 개수를 지정합니다.
# feature_fraction: 각 트리 학습 시 사용할 특징량의 비율을 지정합니다. 과적합을 줄이기 위해 사용됩니다.
# bagging_fraction: 각 트리 학습 시 사용할 데이터 샘플링 비율을 지정합니다. 과적합을 줄이기 위해 사용됩니다.
# bagging_freq: 배깅을 수행할 빈도를 지정합니다.
# min_child_samples: 각 리프 노드에 필요한 최소한의 데이터 샘플 수를 지정합니다. 과적합을 줄이기 위해 사용됩니다.

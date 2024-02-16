import numpy as np
import pandas as pd
from keras.models import Sequential, load_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler, Normalizer, RobustScaler
from sklearn.metrics import accuracy_score, f1_score
from lightgbm import LGBMClassifier, Booster
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

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

best_accuracy = 0.0
best_model = None
import random
while True:
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, random_state=367, stratify=y, shuffle=True)

    scaler = StandardScaler()
    scaler.fit(x_train)
    try:
        x_train = scaler.transform(x_train)
        x_test = scaler.transform(x_test)
        test = scaler.transform(test)
    except ValueError as e:
        print("ValueError occurred during scaling:", e)
    r = random.randint(1, 1000)
    min_num_leaves = 2 
    r2 = random.randint(min_num_leaves, 50)
    r3 = random.randint(2, 50)
    r4 = random.randint(2, 50)
    r5 = random.randint(2, 50)
    r6 = random.uniform(0.000000000001, 1)
    r7 = random.uniform(0.000000000001, 1)
    r8 = random.uniform(0.1, 1)
    r9 = random.uniform(0.000000000001, 1)
    
    
    # 모델 생성 및 학습
    lgbm_params = {"objective": "multiclass",
                   "metric": "multi_logloss",
                   "verbosity": -1,
                   "boosting_type": "gbdt",
                   "random_state": r,
                   "num_class": 7,
                   "learning_rate": r6,# 0.01386432121252535,
                   "n_estimators": 486,
                   "feature_pre_filter": False,
                   "lambda_l1": 1.2149501037669967e-07,
                   "lambda_l2": r7,#0.9230890143196759,
                   "num_leaves": r2,
                   "feature_fraction": r9,
                   "bagging_fraction":r8,# 0.5523862448863431,
                   "bagging_freq": 4,
                   "min_child_samples": r3,
                   "max_depth": r4,
                   "min_samples_leaf":  r5,
                   'n_jobs': -1
                   }


    model = LGBMClassifier(**lgbm_params)
    model.fit(x_train, y_train)

    # 테스트 데이터 예측 및 저장
    y_pred = model.predict(x_test)

    # 정확도 평가
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    print("r",r)    
    print("r2",r2)    
    print("r3",r3)    
    print("r4",r4)    
    print("r5",r5)    
    print("r5",r6)    
    print("r5",r7)    
    print("r5",r8)    
    print("r5",r6)    
    
    # If the current model is the best so far, save it
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model

    if best_accuracy >= 0.928:
        break
# print("Accuracy:", accuracy)
# print("r",r)

# Save the best model
best_model.booster_.save_model("c:/_data/_save/비만33.h5")

# Use the best model to predict on the test data
y_submit = best_model.predict(test)
sample['NObeyesdad'] = y_submit
sample.to_csv(path + "비만33.csv", index=False)

'''
Accuracy: 0.9229287090558767        베스트뭐시기
r 32
r2 367
'''


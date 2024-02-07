import numpy as np
import pandas as pd
from keras.models import Sequential, load_model
from keras.layers import Dense,Dropout,BatchNormalization, AveragePooling2D, Flatten, Conv2D, LSTM, Bidirectional,Conv1D
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler, Normalizer, RobustScaler
from sklearn.metrics import accuracy_score, f1_score
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler
import lightgbm as lgb
#1.데이터
path= "c:\_data\dacon\dechul\\"
train_csv=pd.read_csv(path+"train.csv",index_col=0)
test_csv=pd.read_csv(path+"test.csv",index_col=0)
sample_csv=pd.read_csv(path+"sample_submission.csv")
# train_csv = train_csv[train_csv['총상환이자'] != 0.0]
x= train_csv.drop(['대출등급','최근_2년간_연체_횟수','총연체금액','연체계좌수'],axis=1)
y= train_csv['대출등급']
test_csv = test_csv.drop(['최근_2년간_연체_횟수','총연체금액','연체계좌수'],axis=1)

# print(x.shape)
# print(x.shape)      #  (96294, 14)
# print(test_csv.shape)       #   (64197, 13)
# print(sample_csv.shape)  #    (64197, 2)
# print(np.unique(y,return_counts=True))


y=y.values.reshape(-1,1)

ohe = OneHotEncoder(sparse=False)
ohe = OneHotEncoder()
y_ohe = ohe.fit_transform(y).toarray()

# print(y_ohe,y_ohe.shape)


lb=LabelEncoder()
lb.fit(x['대출기간'])
x['대출기간'] = lb.transform(x['대출기간'])
lb.fit(x['근로기간'])
x['근로기간'] = lb.transform(x['근로기간'])
lb.fit(x['주택소유상태'])
x['주택소유상태'] = lb.transform(x['주택소유상태'])
lb.fit(x['대출목적'])
x['대출목적'] = lb.transform(x['대출목적'])

lb.fit(test_csv['대출기간'])
test_csv['대출기간'] =lb.transform(test_csv['대출기간'])

lb.fit(test_csv['근로기간'])
test_csv['근로기간'] =lb.transform(test_csv['근로기간'])

lb.fit(test_csv['주택소유상태'])
test_csv['주택소유상태'] =lb.transform(test_csv['주택소유상태'])

lb.fit(test_csv['대출목적'])
test_csv['대출목적'] =lb.transform(test_csv['대출목적'])


x_train,x_test,y_train,y_test=train_test_split(x,y_ohe,train_size=0.85,random_state=3,stratify=y_ohe)

# LightGBM 모델 정의 및 학습
params = {
    'objective': 'multiclass',
    'num_class': 4,
    'metric': 'multi_logloss',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
}

train_data = lgb.Dataset(x_train, label=y_train)
test_data = lgb.Dataset(x_test, label=y_test, reference=train_data)

num_round = 100
bst = lgb.train(params, train_data, num_round, valid_sets=[test_data], valid_names=['test'], early_stopping_rounds=10)

# 결과 예측
y_pred = bst.predict(x_test, num_iteration=bst.best_iteration)
y_pred_max = np.argmax(y_pred, axis=1)

# 정확도 및 F1-score 계산
accuracy = np.mean(y_pred_max == y_test)
f1 = f1_score(y_test, y_pred_max, average='macro')

print("Accuracy:", accuracy)
print("F1-score:", f1)

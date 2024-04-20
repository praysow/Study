import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.utils import all_estimators
import warnings
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')

#1.데이터
path= "c:\_data\dacon\dechul\\"
train=pd.read_csv(path+"train.csv",index_col=0)
test=pd.read_csv(path+"test.csv",index_col=0)
sample=pd.read_csv(path+"sample_submission.csv")
x= train.drop(['대출등급', '최근_2년간_연체_횟수', '총연체금액', '연체계좌수'],axis=1)
test= test.drop(['최근_2년간_연체_횟수', '총연체금액', '연체계좌수'],axis=1)
test['대출목적'] = test['대출목적'].replace('결혼', '휴가')
y= train['대출등급']

z = test[test['대출목적'].str.contains('결혼')]

# y = y.reshape(-1,1)
# ohe = OneHotEncoder()
# y = ohe.fit_transform(y).toarray()
# lb = LabelEncoder()

# le = LabelEncoder()
# y = le.fit_transform(train['대출목적'])

# Label Encoding
lb = LabelEncoder()
columns_to_encode = ['대출기간', '근로기간', '주택소유상태', '대출목적']

for column in columns_to_encode:
    x[column] = lb.fit_transform(x[column])
    test[column] = lb.transform(test[column])

y = lb.fit_transform(train['대출등급'])
# 데이터 스케일링
scaler = MinMaxScaler()
scaler = StandardScaler()
scaler.fit(x)
x = scaler.transform(x)

# 데이터 분할
x = x[:-30]
y = y[:-30]

x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.9,random_state=3,stratify=y)   #stratify=y의 비율대로 잘라라

from imblearn.over_sampling import SMOTE

smote= SMOTE(random_state=1,k_neighbors=3)
x_train,y_train = smote.fit_resample(x_train,y_train)

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

model = RandomForestClassifier()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

r2 = accuracy_score(y_test, y_pred)
print("R2 Score:", r2)
'''
R2 Score: 0.8559260413420587
'''
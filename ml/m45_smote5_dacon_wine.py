import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

path = 'c:/_data/dacon/wine/'
train = pd.read_csv(path + 'train.csv', index_col=0)
test = pd.read_csv(path + 'test.csv', index_col=0)
submit = pd.read_csv(path + 'sample_submission.csv')

x = train.drop(['quality'], axis=1)
y = train['quality']
y -= 3
lb = LabelEncoder()
lb.fit(x['type'])
x['type'] = lb.transform(x['type'])
test['type'] = lb.transform(test['type'])

x = x[:-30]
y = y[:-30]

x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.9,random_state=8,stratify=y)   #stratify=y의 비율대로 잘라라

from imblearn.over_sampling import SMOTE

smote= SMOTE(random_state=8,k_neighbors=3)
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
R2 Score: 0.6416819012797075
'''
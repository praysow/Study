import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
path = 'c:/_data/dacon/wine/'
train = pd.read_csv(path + 'train.csv', index_col=0)
test = pd.read_csv(path + 'test.csv', index_col=0)
submit = pd.read_csv(path + 'sample_submission.csv')

lb = LabelEncoder()
lb.fit(train['type'])
train['type'] = lb.transform(train['type'])
test['type'] = lb.transform(test['type'])

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
    
outliers_loc = outliers(train)
train = train.drop(outliers_loc[0])

x = train.drop(['quality'], axis=1)
y = train['quality']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, random_state=8)

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

model = RandomForestClassifier()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

r2 = accuracy_score(y_test, y_pred)
print("R2 Score:", r2)

##### 그래프 그리기
#1. value_count 사용 x
#2. np.unique 사용 x

####3. groupby count() 사용
#plt.bar로 그린다.(quality 컬럼)\
quality_counts = train.groupby('quality').size()
quality_counts.plot(kind='bar')
plt.xlabel('Quality')
plt.ylabel('Count')
plt.title('Distribution of Quality')
plt.show()
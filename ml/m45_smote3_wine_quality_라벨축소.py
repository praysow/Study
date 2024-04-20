import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.ensemble import RandomForestClassifier
datasets=load_wine()
x= datasets.data
y= datasets.target

y = y.copy()

for i,v in enumerate(y):
    if v<=4:
        y[i] = 0
    elif v==5:
        y[i] = 1
    elif v==6:
        y[i] = 2
    elif v==7:
        y[i] = 3
    # elif v==8:
    #     y[i] = 4
    # else:
    #     y[i] = 2
# print(y['quality'].value_counts().sort_index())

# y -= 3
# lb = LabelEncoder()
# lb.fit(x['type'])
# x['type'] = lb.transform(x['type'])

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, random_state=8)

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

model = RandomForestClassifier()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

r2 = f1_score(y_test, y_pred,average='macro')
print("R2 Score:", r2)
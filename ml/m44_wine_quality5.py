import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score,f1_score
from sklearn.ensemble import RandomForestClassifier

path = 'c:/_data/dacon/wine/'
train = pd.read_csv(path + 'train.csv', index_col=0)
test = pd.read_csv(path + 'test.csv', index_col=0)
submit = pd.read_csv(path + 'sample_submission.csv')

# # print(train['quality'].value_counts().sort_index())
# x = train.drop(['quality'], axis=1)
# y = train['quality']
# y = y.copy()

# for i,v in enumerate(y):
#     if v<=4:
#         y[i] = 0
#     elif v==5:
#         y[i] = 1
#     elif v==6:
#         y[i] = 2
#     elif v==7:
#         y[i] = 3
#     elif v==8:
#         y[i] = 4
#     else:
#         y[i] = 2
# # print(y['quality'].value_counts().sort_index())

# # y -= 3
# lb = LabelEncoder()
# lb.fit(x['type'])
# x['type'] = lb.transform(x['type'])
# test['type'] = lb.transform(test['type'])

# x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, random_state=8)

# scaler = StandardScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)

# model = RandomForestClassifier()
# model.fit(x_train, y_train)

# y_pred = model.predict(x_test)

# r2 = f1_score(y_test, y_pred,average='macro')
# print("R2 Score:", r2)


##원판하고 라벨을 변경후

x = train.drop(['quality'], axis=1)
y = train['quality']
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
lb = LabelEncoder()
lb.fit(x['type'])
x['type'] = lb.transform(x['type'])
test['type'] = lb.transform(test['type'])

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


# accs = []

# for i in range(2, 7):
#     train_subset = train
#     for _ in range(i):
#         train_subset = train_subset[train_subset['quality'] != train_subset['quality'].max()]
#     print(f"{i}개:", train_subset['quality'].value_counts().sort_index())

#     x = train.drop(['quality'], axis=1)
#     y = train['quality']
#     y -= 3

#     # 'type' 열의 값 확인
#     print(train['type'].value_counts())

#     # 'type' 열의 값을 LabelEncoder를 사용하여 정수형으로 변환
#     lb = LabelEncoder()
#     x['type'] = lb.fit_transform(x['type'])
#     # test['type'] = lb.transform(test['type'])

#     x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, random_state=8)

#     scaler = StandardScaler()
#     scaler.fit(x_train)
#     x_train = scaler.transform(x_train)
#     x_test = scaler.transform(x_test)

#     model = RandomForestClassifier()
#     model.fit(x_train, y_train)

#     y_pred = model.predict(x_test)

#     acc = accuracy_score(y_test, y_pred)
#     accs.append(acc)

# print("2개:", accs[0])
# print("3개:", accs[1])
# print("4개:", accs[2])
# print("5개:", accs[3])
# print("6개:", accs[4])

'''
2개: 0.7163636363636363
3개: 0.7072727272727273
4개: 0.7127272727272728
5개: 0.7072727272727273
6개: 0.6872727272727273
'''
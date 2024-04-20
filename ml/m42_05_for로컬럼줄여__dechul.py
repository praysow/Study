import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.utils import all_estimators
import warnings
from sklearn.metrics import accuracy_score,log_loss
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, HalvingGridSearchCV, HalvingRandomSearchCV
from sklearn.model_selection import KFold, StratifiedKFold
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
# train.drop(train.index[34488], inplace=True)
y= train['대출등급']

z = test[test['대출목적'].str.contains('결혼')]
# print(z.value_counts)
# print(np.unique(z,return_counts=True))

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

# One-hot Encoding
# ohe = OneHotEncoder(sparse=False)
# y_encoded = ohe.fit_transform(y.values.reshape(-1, 1))

y = lb.fit_transform(train['대출등급'])
# 데이터 스케일링
scaler = MinMaxScaler()
scaler = StandardScaler()
scaler.fit(x)
x = scaler.transform(x)

# 데이터 분할
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.85, random_state=100)

from sklearn.preprocessing import StandardScaler,MinMaxScaler
from xgboost import XGBClassifier
scaler=MinMaxScaler()
scaler=StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
parameters = {
    'n_estimators' : 4000,
    'learning_rate' : 0.2,  #훈련량
    'max_depth' : 3,        #트리 노드의 깊이
    'gamma' : 4,
    'min_child_weight' : 0.01,
    'subsample' : 0.1,      # = dropout
    'colsample_bytree' : 1,
    'colsample_bylevel' : 1,
    'colsample_bynode' : 1,
    'reg_alpha' : 1,        # L1,L2 가중치 규제
    'reg_lambda' : 1,
    }
#2. 모델
model = XGBClassifier()
model.set_params(early_stopping_rounds=10,**parameters)
#3. 훈련
model.fit(x_train,y_train,
          eval_set=[(x_train,y_train),(x_test,y_test)],
          verbose =500,
          eval_metric='mlogloss'
          )
#4. 평가,예측
result = model.score(x_test,y_test)
print("최종점수:",result)
y_pred = model.predict(x_test)
acc = accuracy_score(y_test,y_pred)
print("acc:",acc)

#############
# print(model.feature_importances_)
#for문을 사용해서 피처가 약한놈부터 하나씩 제거
#30,29,28,27....1
# 초기 평가
initial_loss = log_loss(y_test, model.predict_proba(x_test))
initial_accuracy = accuracy_score(y_test, model.predict(x_test))
print(f"Initial Log Loss: {initial_loss}, Initial Accuracy: {initial_accuracy}")

# Feature Importance를 이용한 피처 제거 및 평가
feature_importances = model.feature_importances_
sorted_indices = np.argsort(feature_importances)[::-1]

results = []

for i in range(len(sorted_indices)):
    # i개의 피처를 제거한 새로운 특징 배열 만들기
    reduced_x_train = np.delete(x_train, sorted_indices[:i+1], axis=1)
    reduced_x_test = np.delete(x_test, sorted_indices[:i+1], axis=1)
    
    # 만약 특징의 수가 0이 되면 반복문을 중단
    if reduced_x_train.shape[1] == 0:
        print("No features left to remove.")
        break
    
    # 모델 훈련
    model.fit(reduced_x_train, y_train,
              eval_set=[(reduced_x_train, y_train), (reduced_x_test, y_test)],
              verbose=500,
              eval_metric='mlogloss'
              )
    
    # 피처 제거 후 모델 평가
    logloss = log_loss(y_test, model.predict_proba(reduced_x_test))
    accuracy = accuracy_score(y_test, model.predict(reduced_x_test))
    results.append((i+1, logloss, accuracy))

# 결과 출력
for result in results:
    print(f"After removing top {result[0]} features, Log Loss: {result[1]}, Accuracy: {result[2]}")
    
'''
After removing top 1 features, Log Loss: 0.7832208619751861, Accuracy: 0.711803392177224
After removing top 2 features, Log Loss: 1.3580535349768426, Accuracy: 0.43918310834198687
After removing top 3 features, Log Loss: 1.4701540850804524, Accuracy: 0.363447559709242
After removing top 4 features, Log Loss: 1.4891841659814886, Accuracy: 0.3524402907580478
After removing top 5 features, Log Loss: 1.5308643350300886, Accuracy: 0.3383869851159571
After removing top 6 features, Log Loss: 1.5633271603188632, Accuracy: 0.3226029768085843
After removing top 7 features, Log Loss: 1.5877585877472713, Accuracy: 0.30965732087227416
After removing top 8 features, Log Loss: 1.5898961113327452, Accuracy: 0.3136725510557286
After removing top 9 features, Log Loss: 1.5928072921163081, Accuracy: 0.30834198684665975
'''
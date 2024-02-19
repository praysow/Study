import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.utils import all_estimators
import warnings
from sklearn.metrics import accuracy_score
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

# 교차 검증
n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)

# 모델 및 파라미터 정의
xgb = XGBClassifier(random_state=123)
parameters = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.1, 0.2, 0.3],
    'max_depth': [None, 2, 3, 4, 5],
    'gamma': [0, 1, 2, 3, 4],
    'min_child_weight': [0, 0.001, 0.01, 0.1, 0.5, 1, 5, 10, 100],
    'subsample': [0, 0.1, 0.2, 0.3, 0.5, 1],
    'colsample_bytree': [0, 0.1, 0.2, 0.3, 1],
    'colsample_bylevel': [0, 0.1, 0.2, 0.3, 1],
    'colsample_bynode': [0, 0.1, 0.2, 0.3, 1],
    'reg_alpha': [0, 0.1, 0.01, 0.001, 1, 2, 10],
    'reg_lambda': [0, 0.1, 0.01, 0.001, 1, 2, 10],
    'random_state': [123],
}

# 모델 피팅 및 평가
model = RandomizedSearchCV(xgb, parameters, cv=kfold, n_jobs=22)
model.fit(x_train, y_train)
results = model.score(x_test, y_test)
y_pred = model.predict(x_test)
print("최상의 매개변수 :", model.best_estimator_)
print("최상의 매개변수 :", model.best_params_)
print("최상의 점수 :", model.best_score_)
print("result", results)

'''
최상의 매개변수 : XGBClassifier(base_score=None, booster=None, callbacks=None,
              colsample_bylevel=0.1, colsample_bynode=0.1, colsample_bytree=0.3,
              device=None, early_stopping_rounds=None, enable_categorical=False,
              eval_metric=None, feature_types=None, gamma=2, grow_policy=None,
              importance_type=None, interaction_constraints=None,
              learning_rate=0.2, max_bin=None, max_cat_threshold=None,
              max_cat_to_onehot=None, max_delta_step=None, max_depth=4,
              max_leaves=None, min_child_weight=100, missing=nan,
              monotone_constraints=None, multi_strategy=None, n_estimators=300,
              n_jobs=None, num_parallel_tree=None, objective='multi:softprob', ...)
최상의 매개변수 : {'subsample': 1, 'reg_lambda': 1, 'reg_alpha': 0.001, 'random_state': 123, 'n_estimators': 300, 'min_child_weight': 100, 'max_depth': 4, 'learning_rate': 0.2, 'gamma': 2, 'colsample_bytree': 0.3, 'colsample_bynode': 0.1, 'colsample_bylevel': 0.1}
최상의 점수 : 0.6151815679719695
result 0.616337833160263
'''

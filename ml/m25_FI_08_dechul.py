import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor,GradientBoostingClassifier
from sklearn.utils import all_estimators
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
#1.데이터
path= "c:\_data\dacon\dechul\\"
train_csv=pd.read_csv(path+"train.csv",index_col=0)
test_csv=pd.read_csv(path+"test.csv",index_col=0)
sample_csv=pd.read_csv(path+"sample_submission.csv")
x= train_csv.drop(['대출등급'],axis=1)
y= train_csv['대출등급']
feature_names =  train_csv.columns

print(x.columns)
print(x.shape)
y=y.values.reshape(-1,1)

ohe = OneHotEncoder(sparse=False)
ohe = OneHotEncoder()
y_ohe = ohe.fit_transform(y).toarray()

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


x_train,x_test,y_train,y_test=train_test_split(x,y_ohe,train_size=0.85,random_state=100 ,
                                              #  stratify=y
                                               )
from xgboost import XGBClassifier


allAlgorithms = [
    ('LogisticRegression', XGBClassifier),
    # ('KNeighborsClassifier',GradientBoostingClassifier),
    ('DecisionTreeClassifier', DecisionTreeClassifier),
    ('RandomForestClassifier', RandomForestClassifier)
]
# 피쳐 임포턴스를 사용하여 하위 25%의 피쳐를 선택하는 함수
def select_features_with_importance(model, x_train, feature_names):
    # 모델 피쳐 임포턴스 추출
    feature_importances = model.feature_importances_
    # 피쳐 중요도 기준으로 정렬
    sorted_indices = np.argsort(feature_importances)
    # 하위 25%의 피쳐 인덱스 선택
    num_features_to_keep = int(len(feature_names) * 0.25)  # 상위 75% 선택
    selected_indices = sorted_indices[num_features_to_keep:]
    # 선택된 피쳐 이름 반환
    selected_feature_names = [feature_names[i] for i in selected_indices]
    return selected_feature_names

# 피쳐 임포턴스를 사용하여 하위 25%의 피쳐를 선택하여 모델을 훈련/평가하는 함수
def train_and_evaluate_with_feature_selection(algorithm, x_train, y_train, x_test, y_test, feature_names):
    # 모델 생성
    model = algorithm()
    # 모델 훈련
    model.fit(x_train, y_train)
    # 선택된 피쳐들을 추출
    selected_feature_names = select_features_with_importance(model, x_train, feature_names)
    if len(selected_feature_names) == 0:
        print("선택된 피쳐가 없습니다.")
        return
    # 선택된 피쳐들을 이용하여 새로운 데이터셋 생성
    x_train_selected = x_train[selected_feature_names]
    x_test_selected = x_test[selected_feature_names]
    # 모델 재훈련
    model.fit(x_train_selected, y_train)
    # 모델 평가
    acc = model.score(x_test_selected, y_test)
    print(f"{algorithm.__name__}'s 정확도 (하위 25% 피쳐 선택):", acc)

# 각 알고리즘에 대해 피쳐 임포턴스를 사용하여 하위 25%의 피쳐를 선택하여 모델을 훈련/평가
for name, algorithm in allAlgorithms:
    train_and_evaluate_with_feature_selection(algorithm, x_train, y_train, x_test, y_test, feature_names)

for name, algorithm in allAlgorithms:
    model = algorithm()
    model.fit(x_train, y_train)
    acc = model.score(x_test, y_test)
    print(name, '의 정확도:', acc)
'''
XGBClassifier's 정확도 (하위 25% 피쳐 선택): 0.7940463828314296
DecisionTreeClassifier's 정확도 (하위 25% 피쳐 선택): 0.8349601938386985
RandomForestClassifier's 정확도 (하위 25% 피쳐 선택): 0.7643475250951887
LogisticRegression 의 정확도: 0.7978539286950502
DecisionTreeClassifier 의 정확도: 0.8352371062651437
RandomForestClassifier 의 정확도: 0.6676358601592246
'''
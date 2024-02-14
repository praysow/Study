import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
#1. 데이터
path= "c:\_data\dacon\wine\\"
train_csv = pd.read_csv(path+"train.csv",index_col=0)
test_csv = pd.read_csv(path+"test.csv",index_col=0)
sampleSubmission_csv = pd.read_csv(path+"sample_Submission.csv")
x= train_csv.drop(['quality'], axis=1)
y= train_csv['quality']

lb=LabelEncoder()
lb.fit(x['type'])
x['type'] =lb.transform(x['type'])
test_csv['type'] =lb.transform(test_csv['type'])

feature_names =  train_csv.columns
y -= 3

x_train,x_test,y_train,y_test=train_test_split(x,y, train_size= 0.9193904973982694, random_state=1909,
                                            stratify=y)
from xgboost import XGBClassifier


allAlgorithms = [
    ('LogisticRegression', XGBClassifier),
    ('KNeighborsClassifier',GradientBoostingClassifier),
    ('DecisionTreeClassifier', DecisionTreeClassifier),
    ('RandomForestClassifier', RandomForestClassifier)
]

# 피처 중요도를 사용하여 하위 25%의 피처를 삭제하는 함수
def remove_features_with_low_importance(model, x_train, feature_names):
    # 모델 피처 임포턴스 추출
    feature_importances = model.feature_importances_
    # 피처 중요도 기준으로 정렬
    sorted_indices = np.argsort(feature_importances)
    # 하위 25%의 피쳐 인덱스 선택
    num_features_to_remove = int(len(feature_names) * 0.25)  # 하위 25% 선택
    features_to_remove = sorted_indices[:num_features_to_remove]
    # 선택된 피처 삭제
    selected_feature_names = [feature_names[i] for i in range(len(feature_names)) if i not in features_to_remove]
    return selected_feature_names

# 피처 중요도를 사용하여 하위 25%의 피처를 삭제하여 모델을 훈련/평가하는 함수
def train_and_evaluate_with_feature_removal(algorithm, x_train, y_train, x_test, y_test, feature_names):
    # 모델 생성
    model = algorithm()
    # 모델 훈련
    model.fit(x_train, y_train)
    # 선택된 피처들을 추출
    selected_feature_names = remove_features_with_low_importance(model, x_train, feature_names)
    if len(selected_feature_names) == 0:
        print("선택된 피처가 없습니다.")
        return
    # 선택된 피처들이 실제 데이터프레임의 열로 존재하는지 확인
    missing_features = [feature for feature in selected_feature_names if feature not in x_train.columns]
    if missing_features:
        print("다음 피처가 데이터프레임에 존재하지 않습니다:", missing_features)
        return
    # 선택된 피처들을 이용하여 새로운 데이터셋 생성
    x_train_selected = x_train[selected_feature_names]
    x_test_selected = x_test[selected_feature_names]
    # 모델 재훈련
    model.fit(x_train_selected, y_train)
    # 모델 평가
    acc = model.score(x_test_selected, y_test)
    print(f"{algorithm.__name__}'s 정확도 (하위 25% 피처 삭제):", acc)


for name, algorithm in allAlgorithms:
    train_and_evaluate_with_feature_removal(algorithm, x_train, y_train, x_test, y_test, feature_names)

for name, algorithm in allAlgorithms:
    model = algorithm()
    model.fit(x_train, y_train)
    acc = model.score(x_test, y_test)
    print(name, '의 정확도:', acc)

'''
XGBClassifier's 정확도 (하위 25% 피처 삭제): 0.6734234234234234
GradientBoostingClassifier's 정확도 (하위 25% 피처 삭제): 0.581081081081081
다음 피처가 데이터프레임에 존재하지 않습니다: ['quality']
RandomForestClassifier's 정확도 (하위 25% 피처 삭제): 0.6981981981981982
LogisticRegression 의 정확도: 0.6599099099099099
KNeighborsClassifier 의 정확도: 0.6261261261261262
DecisionTreeClassifier 의 정확도: 0.5878378378378378
RandomForestClassifier 의 정확도: 0.6914414414414415
'''
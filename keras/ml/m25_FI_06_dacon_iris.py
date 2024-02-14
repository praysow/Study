import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier

#1.데이터
path= "c:\_data\dacon\iris\\"
train_csv = pd.read_csv(path+"train.csv",index_col=0)
test_csv = pd.read_csv(path+"test.csv",index_col=0)
sampleSubmission_csv = pd.read_csv(path+"sample_Submission.csv")
x= train_csv.drop(['species'], axis=1)
y= train_csv['species']
feature_names =  train_csv.columns

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8,
                                                    random_state=1,         #850:acc=1
                                                    stratify=y              #stratify는 분류에서만 사용
                                                    )
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

# 피쳐 중요도를 사용하여 하위 25%의 피쳐를 삭제하여 모델을 훈련/평가하는 함수
def train_and_evaluate_with_feature_removal(algorithm, x_train, y_train, x_test, y_test, feature_names):
    # 모델 생성
    model = algorithm()
    # 모델 훈련
    model.fit(x_train, y_train)
    # 선택된 피쳐들을 추출
    selected_feature_names = remove_features_with_low_importance(model, x_train, feature_names)
    if len(selected_feature_names) == 0:
        print("선택된 피쳐가 없습니다.")
        return
    # 'species' 열이 포함되어 있는지 확인하고 제거
    if 'species' in selected_feature_names:
        selected_feature_names.remove('species')
    # 선택된 피쳐들을 이용하여 새로운 데이터셋 생성
    x_train_selected = x_train[selected_feature_names]
    x_test_selected = x_test[selected_feature_names]
    # 모델 재훈련
    model.fit(x_train_selected, y_train)
    # 모델 평가
    acc = model.score(x_test_selected, y_test)
    print(f"{algorithm.__name__}'s 정확도 (하위 25% 피쳐 삭제):", acc)



# 각 알고리즘에 대해 피쳐 중요도를 사용하여 하위 25%의 피쳐를 삭제하여 모델을 훈련/평가
for name, algorithm in allAlgorithms:
    train_and_evaluate_with_feature_removal(algorithm, x_train, y_train, x_test, y_test, feature_names)

for name, algorithm in allAlgorithms:
    model = algorithm()
    model.fit(x_train, y_train)
    acc = model.score(x_test, y_test)
    print(name, '의 정확도:', acc)
'''
XGBClassifier's 정확도 (하위 25% 피쳐 삭제): 0.9166666666666666
GradientBoostingClassifier's 정확도 (하위 25% 피쳐 삭제): 0.9583333333333334
DecisionTreeClassifier's 정확도 (하위 25% 피쳐 삭제): 0.9583333333333334
RandomForestClassifier's 정확도 (하위 25% 피쳐 삭제): 0.9583333333333334
LogisticRegression 의 정확도: 0.9166666666666666
KNeighborsClassifier 의 정확도: 0.9583333333333334
DecisionTreeClassifier 의 정확도: 0.9166666666666666
RandomForestClassifier 의 정확도: 0.9583333333333334
'''
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import RobustScaler

# 데이터 불러오기
path = "c:\_data\dacon\cancer\\"
train_csv = pd.read_csv(path + "train.csv", index_col=0)

# 필요한 피처 선택 및 타겟 설정
x = train_csv.drop(['Outcome', 'Pregnancies', 'Glucose'], axis=1)
y = train_csv['Outcome']

# 피처 이름 업데이트
feature_names = x.columns

# 데이터 스케일링
scaler = RobustScaler()
x = scaler.fit_transform(x)

# 데이터 분할
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=1, stratify=y)

# 분류 모델 정의
allAlgorithms = [
    ('LogisticRegression', XGBClassifier),
    ('KNeighborsClassifier', GradientBoostingClassifier),
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

# 피처 중요도를 사용하여 하위 25%의 피쳐를 삭제하여 모델을 훈련/평가하는 함수
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
    # 선택된 피쳐들을 이용하여 새로운 데이터셋 생성
    x_train_selected = x_train[:, [feature_names.get_loc(feature) for feature in selected_feature_names]]
    x_test_selected = x_test[:, [feature_names.get_loc(feature) for feature in selected_feature_names]]
    # 모델 재훈련
    model.fit(x_train_selected, y_train)
    # 모델 평가
    acc = model.score(x_test_selected, y_test)
    print(f"{algorithm.__name__}'s 정확도 (하위 25% 피쳐 삭제):", acc)

# 각 알고리즘에 대해 피쳐 중요도를 사용하여 하위 25%의 피쳐를 삭제하여 모델을 훈련/평가
for name, algorithm in allAlgorithms:
    train_and_evaluate_with_feature_removal(algorithm, x_train, y_train, x_test, y_test, feature_names)

'''
XGBClassifier's 정확도 (하위 25% 피쳐 삭제): 0.6717557251908397
GradientBoostingClassifier's 정확도 (하위 25% 피쳐 삭제): 0.732824427480916
RandomForestClassifier's 정확도 (하위 25% 피쳐 삭제): 0.7251908396946565
'''
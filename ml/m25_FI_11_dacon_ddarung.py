import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import Perceptron,LogisticRegression,SGDClassifier
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor,GradientBoostingRegressor
from sklearn.utils import all_estimators
import warnings
from sklearn.metrics import accuracy_score
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV,HalvingGridSearchCV,HalvingRandomSearchCV
warnings.filterwarnings('ignore')
import time
#1. 데이터

path= "c:\_data\dacon\ddarung\\"

train_csv = pd.read_csv(path+"train.csv",index_col=0)
test_csv= pd.read_csv(path+"test.csv",index_col=0)
submission_csv= pd.read_csv(path+"submission.csv")

train_csv=train_csv.fillna(train_csv.mean())                         #test는 dropna를 하면 안되고 결측치를 변경해줘야한다
# train_csv=train_csv.fillna(0)
test_csv=test_csv.fillna(test_csv.mean())                         #test는 dropna를 하면 안되고 결측치를 변경해줘야한다
# test_csv=test_csv.fillna(0)

x= train_csv.drop(['count','hour', 'hour_bef_temperature', 'hour_bef_precipitation'],axis=1)
y= train_csv['count']

print(x.columns)
print(x.shape)

feature_names =  train_csv.columns

x_train,x_test,y_train,y_test=train_test_split(x,y, train_size=0.8, random_state=6)

from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer, RobustScaler
scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

from xgboost import XGBRegressor


allAlgorithms = [
    ('LogisticRegression', XGBRegressor),
    ('KNeighborsClassifier',GradientBoostingRegressor),
    ('DecisionTreeClassifier', DecisionTreeRegressor),
    ('RandomForestClassifier', RandomForestRegressor)
]

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
    missing_features = [feature for feature in selected_feature_names if feature not in feature_names]
    if missing_features:
        print("다음 피처가 데이터프레임에 존재하지 않습니다:", missing_features)
        return
    # 선택된 피처들을 이용하여 새로운 데이터셋 생성
    x_train_selected = x_train[:, [feature_names.get_loc(feature) for feature in selected_feature_names]]
    x_test_selected = x_test[:, [feature_names.get_loc(feature) for feature in selected_feature_names]]
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


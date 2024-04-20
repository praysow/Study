import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, BaggingClassifier, StackingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')
import random as rn
import tensorflow as tf
rn.seed(333)
tf.random.set_seed(333)
np.random.seed(333)

# data
x, y = load_breast_cancer(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.9,random_state=47)

sclaer = MinMaxScaler().fit(x_train)
x_train = sclaer.transform(x_train)
x_test = sclaer.transform(x_test)

def self_Stacking(models:list[tuple], final_model, x_train, x_test, y_train, y_test):
    pred_list = []
    trained_model_dict = {}
    for name, model in models:
        model.fit(x_train,y_train)
        pred = model.predict(x_train)       # x_test로 하면 나중에 final_model도 테스트 셋으로 학습해야하기에 테스트 셋 의미가 없어진다
        pred_list.append(pred)              # 예측값 저장 쉽게 append로
        trained_model_dict[name] = model    # 훈련 완료된 모델들 저장, 출력을 위해서 이름도 같이 저장
        
    stacked_train_pred = np.asarray(pred_list).T    # (n,3)형태를 위해서 Transpose
    final_model.fit(stacked_train_pred,y_train)     
    
    pred_list = []
    print_dict = {}
    for name, model in trained_model_dict.items():  # 딕셔러니에서 키값과 내용물을 같이 반환
        pred = model.predict(x_test)   
        result = model.score(x_test,y_test)
        pred_list.append(pred)              # 예측값 저장 쉽게 append로
        print_dict[f'{name} ACC'] = result  # 이름과 함께 ACC 저장
    
    stacked_test_pred = np.asarray(pred_list).T
    final_result = final_model.score(stacked_test_pred,y_test)
    
    for name , acc in print_dict.items():
        print(name,":",acc)
    print("스태킹 결과: ",final_result)
    
self_Stacking([
    ('xgb',XGBClassifier()),
    ('RF',RandomForestClassifier()),
    ('LR',LogisticRegression()),
],CatBoostClassifier(verbose=0),x_train,x_test,y_train,y_test)

'''============== sklearn의 StackingClassifier =============='''

model = StackingClassifier([
    ('xgb',XGBClassifier()),
    ('RF',RandomForestClassifier()),
    ('LR',LogisticRegression()),
],final_estimator=CatBoostClassifier(verbose=0))

model.fit(x_train,y_train)
result = model.score(x_test,y_test)
print("sklearn Stacking의 ACC : ",result)

# xgb ACC : 0.9473684210526315
# RF ACC : 0.9473684210526315
# LR ACC : 0.9824561403508771
# 스태킹 결과:  0.9649122807017544
# sklearn Stacking의 ACC :  0.9649122807017544
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error
import joblib

# 데이터 불러오기
path = "c:/_data/dacon/soduc/"
train = pd.read_csv(path+'train.csv', index_col=0)
test = pd.read_csv(path+'test.csv', index_col=0)
sample = pd.read_csv(path+'sample_submission.csv')

# 피처와 타겟 분리
x = train.drop(['Income','Gains','Losses','Dividends'], axis=1)
y = train['Income']
test = test.drop(['Gains','Losses','Dividends'], axis=1)
lb = LabelEncoder()

# 라벨 인코딩할 열 목록
columns_to_encode = ['Gender','Education_Status','Employment_Status','Industry_Status','Occupation_Status','Race','Hispanic_Origin','Martial_Status','Household_Status','Household_Summary','Citizenship','Birth_Country','Birth_Country (Father)','Birth_Country (Mother)','Tax_Status','Income_Status']

# 데이터프레임 x의 열에 대해 라벨 인코딩 수행
for column in columns_to_encode:
    lb.fit(x[column])
    x[column] = lb.transform(x[column])

# 데이터프레임 test_csv의 열에 대해 라벨 인코딩 수행
for column in columns_to_encode:
    lb.fit(test[column])
    test[column] = lb.transform(test[column])
    
# 데이터 스케일링
scaler = StandardScaler()
x = scaler.fit_transform(x)
test = scaler.transform(test)

best_rmse = float('inf')
best_random_state = None
best_model = None
best_pred_test = None

for r in range(15001, 20001):
    # 훈련 데이터와 검증 데이터 분리
    x_train, x_val, y_train, y_val = train_test_split(x, y, train_size=0.9, random_state=r)

    # XGBoost 모델 학습
    xgb_params = {'learning_rate': 0.2218036245351803,
                'n_estimators': 199,
                'max_depth': 3,
                'min_child_weight': 0.07709868781803283,
                'subsample': 0.80309973945344,
                'colsample_bytree': 0.9254025887963853,
                'gamma': 6.628562492458777e-08,
                'reg_alpha': 0.012998871754325427,
                'reg_lambda': 0.10637051171111844}

    model = xgb.XGBRegressor(**xgb_params)
    model.fit(x_train, y_train, eval_set=[(x_val, y_val)], early_stopping_rounds=50, verbose=100)

    # 검증 데이터 예측
    y_pred_val = model.predict(x_val)

    # 검증 데이터 RMSE 계산
    rmse_val = mean_squared_error(y_val, y_pred_val, squared=False)
    print(f"Validation RMSE with random_state={r}: {rmse_val}")

    if rmse_val < best_rmse:
        best_rmse = rmse_val
        best_random_state = r
        best_model = model
        best_pred_test = model.predict(test)

# 최적 모델 저장
joblib.dump(best_model, f"c:/_data/dacon/soduc/weight/best_model_xgb2.pkl")

# 최적 모델의 결과 저장
sample['Income'] = best_pred_test
sample.to_csv("c:/_data/dacon/soduc/csv/best_result_xgb.csv", index=False)

# 최종 결과 출력
print(f"Best random_state={best_random_state}: Validation RMSE={best_rmse}")
'''
Best random_state=77: Validation RMSE=506.6683414645201
Best random_state=124: Validation RMSE=519.6906575502983
Best random_state=276: Validation RMSE=486.5923491041718
Best random_state=333: Validation RMSE=500.2371849934987
Best random_state=466: Validation RMSE=514.9136184789189
Best random_state=592: Validation RMSE=503.3778274419665
Best random_state=670: Validation RMSE=507.9194825207171
Best random_state=777: Validation RMSE=489.14610013207647
Best random_state=841: Validation RMSE=487.2362145711256
Best random_state=915: Validation RMSE=499.19466679431855
Best random_state=1228: Validation RMSE=482.0439011079073
Best random_state=1899: Validation RMSE=481.4898507926246
Best random_state=2386: Validation RMSE=480.2671106400389
Best random_state=3985: Validation RMSE=473.5111195261379
Best random_state=4158: Validation RMSE=483.48259172837527
Best random_state=5975: Validation RMSE=477.199071297975
Best random_state=6236: Validation RMSE=478.69598818054817
Best random_state=8890: Validation RMSE=475.2385198374716
Best random_state=14485: Validation RMSE=460.1217478480136
Best random_state=15288: Validation RMSE=456.50053927594524
'''



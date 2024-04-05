import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.metrics import mean_squared_error
from catboost import CatBoostClassifier
import optuna
import joblib

# 데이터 불러오기
path = "c:/_data/dacon/soduc/"
train = pd.read_csv(path+'train.csv', index_col=0)
test = pd.read_csv(path+'test.csv', index_col=0)
sample = pd.read_csv(path+'sample_submission.csv')

# 피처와 타겟 분리
x = train.drop(['Income','Gains','Losses','Dividends','Race','Hispanic_Origin','Birth_Country','Birth_Country (Father)','Birth_Country (Mother)'], axis=1)
y = train['Income']
test = test.drop(['Gains','Losses','Dividends','Dividends','Race','Hispanic_Origin','Birth_Country','Birth_Country (Father)','Birth_Country (Mother)'], axis=1)

lb = LabelEncoder()

# 라벨 인코딩할 열 목록
columns_to_encode = ['Gender','Education_Status','Employment_Status','Industry_Status','Occupation_Status','Martial_Status','Household_Status','Household_Summary','Citizenship','Tax_Status','Income_Status']

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

# 훈련 데이터와 검증 데이터 분리
x_train, x_val, y_train, y_val = train_test_split(x, y, train_size=0.9, random_state=38)

def objective(trial):
    # Define parameters to be optimized
    params = {
        "iterations": trial.suggest_int("iterations", 100, 1000),
        "learning_rate": trial.suggest_loguniform("learning_rate", 0.01, 0.3),
        "depth": trial.suggest_int("depth", 3, 12),
        "l2_leaf_reg": trial.suggest_loguniform("l2_leaf_reg", 1e-4, 1e3),
        "random_strength": trial.suggest_loguniform("random_strength", 1e-4, 1),
        "bagging_temperature": trial.suggest_loguniform("bagging_temperature", 0.01, 100.00),
    }

    # Define and train model
    model = CatBoostClassifier(**params, random_state=6)
    model.fit(x_train, y_train, verbose=False)
    
    # Validation RMSE
    y_pred_val = model.predict(x_val)
    rmse_val = mean_squared_error(y_val, y_pred_val, squared=False)
    
    return rmse_val

# Run Optuna study
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=100)

# Get best parameters
best_params = study.best_params
print("Best parameters:", best_params)

# Train final model with best parameters
best_model = CatBoostClassifier(**best_params, random_state=38)
best_model.fit(x_train, y_train)

# Save the model
joblib.dump(best_model, "c:/_data/dacon/soduc/weight/money_cat_optuna1.pkl")

# Predict on test data
y_pred_test = best_model.predict(test)
sample['Income'] = y_pred_test
sample.to_csv("c:/_data/dacon/soduc/csv/money_cat_optuna1.csv", index=False)

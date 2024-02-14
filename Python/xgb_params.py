from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# Iris 데이터셋 로드
iris = load_iris()
x = iris.data
y = iris.target

# 데이터 분할
x_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# XGBClassifier 모델 초기화 및 학습
model = XGBClassifier(
    booster='gbtree',
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    subsample=1.0,
    colsample_bytree=1.0,
    gamma=0,
    reg_alpha=0,
    reg_lambda=1,
    scale_pos_weight=1,
    objective='multi:softmax',
    eval_metric='mlogloss',
    early_stopping_rounds=None,
    verbosity=1,
    random_state=None,
    n_jobs=None
)
model.fit(x_train, y_train)

# 테스트 데이터에 대한 예측
y_pred = model.predict(X_test)

# 정확도 계산
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

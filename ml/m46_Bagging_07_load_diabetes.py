from xgboost import XGBClassifier
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import numpy as np

# 데이터 로드
x, y = load_diabetes(return_X_y=True)

# 데이터 분할
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=3)
from sklearn.preprocessing import StandardScaler,MinMaxScaler
sclaer = MinMaxScaler().fit(x_train)
x_train = sclaer.transform(x_train)
x_test = sclaer.transform(x_test)
from sklearn.ensemble import BaggingRegressor
from sklearn.linear_model import LogisticRegression

# model 
model = BaggingRegressor(LogisticRegression(),
                          n_estimators=100,
                          n_jobs=-1,
                          random_state=47,
                          bootstrap=True,   # default 중복허용
                          )

# fit & pred
model.fit(x_train,y_train,
        #   eval_set=[(x_train,y_train), (x_test,y_test)],
        #   verbose=1,
        #   eval_metric='logloss',
          )

result = model.score(x_test,y_test)
print("Score: ",result)

pred = model.predict(x_test)
acc = r2_score(y_test,pred)
print("ACC: ",acc)
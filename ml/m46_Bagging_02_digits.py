from xgboost import XGBClassifier,XGBRegressor
from sklearn.datasets import load_digits
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split,KFold, StratifiedKFold,GridSearchCV,RandomizedSearchCV,HalvingRandomSearchCV,HalvingGridSearchCV
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import numpy as np
from sklearn.metrics import accuracy_score,log_loss
#1. 데이터
x,y = load_digits(return_X_y=True)

x_train,x_test,y_train,y_test=train_test_split(x,y, train_size=0.8, random_state=3)
sclaer = MinMaxScaler().fit(x_train)
x_train = sclaer.transform(x_train)
x_test = sclaer.transform(x_test)
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression

# model 
model = BaggingClassifier(LogisticRegression(),
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
acc = accuracy_score(y_test,pred)
print("ACC: ",acc)
'''
Score:  0.9638888888888889
ACC:  0.9638888888888889
'''
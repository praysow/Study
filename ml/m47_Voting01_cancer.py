import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler,PolynomialFeatures
from xgboost import XGBClassifier
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
#1.데이터
x,y = load_breast_cancer(return_X_y=True)

x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.9,random_state=1)

sclaer = MinMaxScaler().fit(x_train)
x_train = sclaer.transform(x_train)
x_test = sclaer.transform(x_test)

# model 
from sklearn.ensemble import BaggingClassifier, VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
model = VotingClassifier([
    ('LR',LogisticRegression()),
    ('RF',RandomForestClassifier()),
    ('XGB',XGBClassifier()),
    ], voting='hard')

# fit & pred
model.fit(x_train,y_train,)

result = model.score(x_test,y_test)
print("Score: ",result)

pred = model.predict(x_test)
acc = accuracy_score(y_test,pred)
print("ACC: ",acc)
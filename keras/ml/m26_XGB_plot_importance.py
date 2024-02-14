import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

# 1.데이터
datasets = load_iris()
x = datasets.data
y = datasets['target']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8,
                                                    random_state=1,
                                                    stratify=y
                                                    )


model1 =XGBClassifier()
model2 =GradientBoostingClassifier()
model3 =DecisionTreeClassifier()
model4 =RandomForestClassifier()
model1.fit(x_train,y_train)
model2.fit(x_train,y_train)
model3.fit(x_train,y_train)
model4.fit(x_train,y_train)

# 4.결과예측
result1 = model1.score(x_test,y_test)
y_predict1 = model1.predict(x_test)
result2 = model2.score(x_test,y_test)
y_predict2 = model2.predict(x_test)
result3 = model3.score(x_test,y_test)
y_predict3 = model3.predict(x_test)
result4 = model4.score(x_test,y_test)
y_predict4 = model4.predict(x_test)

print("acc :", result1) #acc : 0.9666666666666667
print("acc :", result2) #acc : 0.9666666666666667
print("acc :", result3) #acc : 0.9666666666666667
print("acc :", result4) #acc : 0.9576

from xgboost.plotting import plot_importance
plot_importance(model1)
plt.show()

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from xgboost import XGBClassifier
# 1.데이터
x,y = load_iris(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8,
                                                    random_state=1,         #850:acc=1
                                                    stratify=y              #stratify는 분류에서만 사용
                                                    )


# model3 = XGBClassifier(random_state = 100)
# model3.fit(x_train,y_train)

# # 4.결과예측
# result3 = model3.score(x_test,y_test)
# y_predict3 = model3.predict(x_test)
# # result4 = model4.score(x_test,y_test)
# # y_predict4 = model4.predict(x_test)

# print("acc :", result3) #acc : 0.9666666666666667
# # print("acc :", result4) #acc : 0.9576'

# print(model3.feature_importances_)
'''
acc : 0.9333333333333333
[0.01776667 0.01016624 0.8795395  0.09252758]
'''

class CustomXGBClassifier(XGBClassifier):
    def __str__(self):
        return 'XGBClassifier()'        #클래스를 불렀을때  XGBCLassifier로 리턴해주겠다
aaa=CustomXGBClassifier
print(aaa)

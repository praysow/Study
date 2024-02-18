import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron,LogisticRegression
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
# 1.데이터
x,y = load_iris(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8,
                                                    random_state=1,         #850:acc=1
                                                    stratify=y              #stratify는 분류에서만 사용
                                                    )


allAlgorithms = [
    ('LogisticRegression', LogisticRegression),
    ('KNeighborsClassifier', KNeighborsClassifier),
    ('DecisionTreeClassifier', DecisionTreeClassifier),
    ('RandomForestClassifier', RandomForestClassifier)
]

# 3. 모델 훈련 및 평가
for name, algorithm in allAlgorithms:
    model = algorithm()
    model.fit(x_train, y_train)
    acc = model.score(x_test, y_test)
    print(name, '의 정확도:', acc)


# model = LinearSVC(C=100)
# 'model1 = LogisticRegression()
# model2 = KNeighborsClassifier()
# model3 = DecisionTreeClassifier()
# model4 = RandomForestClassifier()
# model1.fit(x_train,y_train)
# model2.fit(x_train,y_train)
# model3.fit(x_train,y_train)
# model4.fit(x_train,y_train)

# # 4.결과예측
# result1 = model1.score(x_test,y_test)
# y_predict1 = model1.predict(x_test)
# result2 = model2.score(x_test,y_test)
# y_predict2 = model2.predict(x_test)
# result3 = model3.score(x_test,y_test)
# y_predict3 = model3.predict(x_test)
# result4 = model4.score(x_test,y_test)
# y_predict4 = model4.predict(x_test)

# print("acc :", result1) #acc : 0.9666666666666667
# print("acc :", result2) #acc : 0.9666666666666667
# print("acc :", result3) #acc : 0.9666666666666667
# print("acc :", result4) #acc : 0.9576'


'''
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import Perceptron,LogisticRegression
from sklearn.linear_model import LogisticRegression,LinearRegrssion
from sklearn.neighbors import KNeighborsClassifier,KNighborTegressor
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor

'''
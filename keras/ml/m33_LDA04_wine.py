import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import Perceptron,LogisticRegression,SGDClassifier
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.utils import all_estimators
import warnings
from sklearn.metrics import accuracy_score
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV,HalvingGridSearchCV,HalvingRandomSearchCV
warnings.filterwarnings('ignore')
import time
from sklearn.datasets import load_wine

#1.데이터
datasets= load_wine()
x= datasets.data
y= datasets.target

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda = LinearDiscriminantAnalysis(n_components=1)
x= lda.fit_transform(x,y)
x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.7,random_state=3,shuffle=True,stratify=y)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

ra = np.arange(1, min(x_train.shape) + 1)
# ra = np.arange(1, 20)

for n_components in ra:
    evr = lda.explained_variance_ratio_

    # PCA 모델 학습 및 평가
    model = RandomForestClassifier()
    model.fit(x_train, y_train)
    acc = model.score(x_test, y_test)
    print(f'n_components={n_components}의 정확도:', acc)
    print(evr)
    print(sum(evr))
    evr_cumsum = np.cumsum(evr)      #누적합
    print(evr_cumsum)
import matplotlib.pyplot as plt
plt.plot(evr_cumsum)
plt.grid()
# plt.show()
'''
n_components=1의 정확도: 0.9074074074074074
[0.68747889]
0.6874788878860782
[0.68747889]
'''
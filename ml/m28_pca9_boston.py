from sklearn.datasets import load_boston
datasets = load_boston()
x = datasets.data
y = datasets.target

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
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.9,random_state=100)

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

ra = np.arange(1, min(x_train.shape) + 1)

for n_components in ra:
    pca = PCA(n_components=n_components)
    x_train_pca = pca.fit_transform(x_train)
    x_test_pca = pca.transform(x_test)
    evr = pca.explained_variance_ratio_

    # PCA 모델 학습 및 평가
    model = RandomForestRegressor()
    model.fit(x_train_pca, y_train)
    acc = model.score(x_test_pca, y_test)
    print(f'n_components={n_components}의 정확도:', acc)
    # print(evr)
    # print(sum(evr))
    evr_cumsum = np.cumsum(evr)      #누적합
    print(evr_cumsum)
import matplotlib.pyplot as plt
plt.plot(evr_cumsum)
plt.grid()
plt.show()
'''
n_components=1의 정확도: 0.07019225173099519
[0.47429622]
n_components=2의 정확도: 0.6347551573091608
[0.47429622 0.58272418]
n_components=3의 정확도: 0.8558735356737007
[0.47429622 0.58272418 0.67737999]
n_components=4의 정확도: 0.8383798178821071
[0.47429622 0.58272418 0.67737999 0.74499063]
n_components=5의 정확도: 0.882603572490387
[0.47429622 0.58272418 0.67737999 0.74499063 0.81020132]
n_components=6의 정확도: 0.8909257336909845
[0.47429622 0.58272418 0.67737999 0.74499063 0.81020132 0.86125839]
n_components=7의 정확도: 0.8871129575245446
[0.47429622 0.58272418 0.67737999 0.74499063 0.81020132 0.86125839
 0.89990362]
n_components=8의 정확도: 0.8840658336526646
[0.47429622 0.58272418 0.67737999 0.74499063 0.81020132 0.86125839
 0.89990362 0.92991159]
n_components=9의 정확도: 0.8828893906334649
[0.47429622 0.58272418 0.67737999 0.74499063 0.81020132 0.86125839
 0.89990362 0.92991159 0.95067668]
n_components=10의 정확도: 0.8967010280262688
[0.47429622 0.58272418 0.67737999 0.74499063 0.81020132 0.86125839
 0.89990362 0.92991159 0.95067668 0.96775063]
n_components=11의 정확도: 0.9041623145126124
[0.47429622 0.58272418 0.67737999 0.74499063 0.81020132 0.86125839
 0.89990362 0.92991159 0.95067668 0.96775063 0.98189539]
n_components=12의 정확도: 0.8860254835390267
[0.47429622 0.58272418 0.67737999 0.74499063 0.81020132 0.86125839
 0.89990362 0.92991159 0.95067668 0.96775063 0.98189539 0.99503829]
n_components=13의 정확도: 0.8990598731153951
[0.47429622 0.58272418 0.67737999 0.74499063 0.81020132 0.86125839
 0.89990362 0.92991159 0.95067668 0.96775063 0.98189539 0.99503829
 1.        ]
'''
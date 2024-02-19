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
#1. 데이터

path= "c:\_data\kaggle\\bike\\"
train_csv = pd.read_csv(path+"train.csv",index_col=0)
test_csv = pd.read_csv(path+"test.csv",index_col=0)
sampleSubmission_csv = pd.read_csv(path+"sampleSubmission.csv")

# print(train_csv)
# print(test_csv)
# print(submission_csv)

# print("train",train_csv.shape)      #(10886, 11)
# print("test",test_csv.shape)       #(6493, 8)
# print("sub",sampleSubmission_csv.shape) #(6493, 2)

#train_csv=train_csv.dropna()
# train_csv=train_csv.fillna(train_csv.mean())
# train_csv=train_csv.fillna(0)
# test_csv=test_csv.fillna(test_csv.mean())
#test_csv=test_csv.fillna(0)

x= train_csv.drop(['count','casual','registered'], axis=1)
y= train_csv['count']

x_train,x_test,y_train,y_test=train_test_split(x,y, train_size=0.8, random_state=3)
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
    print(evr)
    # print(sum(evr))
    evr_cumsum = np.cumsum(evr)      #누적합
    # print(evr_cumsum)
import matplotlib.pyplot as plt
plt.plot(evr_cumsum)
plt.grid()
# plt.show()
'''
n_components=1의 정확도: -0.06526387109006349
[0.26520508]
n_components=2의 정확도: 0.19500570470853495
[0.26520508 0.46363353]
n_components=3의 정확도: 0.2903578183714399
[0.26520508 0.46363353 0.62075139]
n_components=4의 정확도: 0.2882090465227234
[0.26520508 0.46363353 0.62075139 0.74817657]
n_components=5의 정확도: 0.29348892116619574
[0.26520508 0.46363353 0.62075139 0.74817657 0.84755726]
n_components=6의 정확도: 0.29047585200281567
[0.26520508 0.46363353 0.62075139 0.74817657 0.84755726 0.94055788]
n_components=7의 정확도: 0.30503347493163724
[0.26520508 0.46363353 0.62075139 0.74817657 0.84755726 0.94055788
 0.99822276]
n_components=8의 정확도: 0.3096730837793533
[0.26520508 0.46363353 0.62075139 0.74817657 0.84755726 0.94055788
 0.99822276 1.        ]
'''
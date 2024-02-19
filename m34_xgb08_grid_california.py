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
import time
from sklearn.datasets import fetch_california_housing

#1.데이터
datasets = fetch_california_housing()
x =datasets.data
y =datasets.target

# print(x)   #(20640, 8)
# print(y)   #(20640,)
# print(x.shape,y.shape)
#print(datasets.feature_names)  #['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
import numpy as np

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.7, random_state=130)
#2. 모델구성
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
n_components=1의 정확도: -0.28839829317407184
[0.25293935]
n_components=2의 정확도: 0.17766975018997577
[0.25293935 0.49001918]
n_components=3의 정확도: 0.3759200387855246
[0.25293935 0.49001918 0.64920341]
n_components=4의 정확도: 0.5605812346239676
[0.25293935 0.49001918 0.64920341 0.77757655]
n_components=5의 정확도: 0.5913978555286821
[0.25293935 0.49001918 0.64920341 0.77757655 0.90304961]
n_components=6의 정확도: 0.65133302615782
[0.25293935 0.49001918 0.64920341 0.77757655 0.90304961 0.98493218]
n_components=7의 정확도: 0.7117765389278309
[0.25293935 0.49001918 0.64920341 0.77757655 0.90304961 0.98493218
 0.99464001]
n_components=8의 정확도: 0.7239565133245911
[0.25293935 0.49001918 0.64920341 0.77757655 0.90304961 0.98493218
 0.99464001 1.        ]
'''